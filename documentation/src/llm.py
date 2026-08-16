"""Language-model client with a deterministic, offline stand-in.

Every surface in this tool — pre-visit brief, discharge summary, plain-English
rewrite — asks a model for a *draft* through one small interface,
``BaseLLM.complete(...)``. The grounding checks, safety checks, and metrics that
sit around that draft live in the other modules and run the same way no matter
which model produced the text.

Three implementations ship here:

- ``AnthropicLLM`` wraps the official ``anthropic`` SDK. It is lazy: importing
  this module never touches the network or needs a key. The client is built the
  first time you actually call it.
- ``MockLLM`` is a deterministic stand-in. It drafts each surface from the input
  alone, with no key and no network, so the whole tool — tests and demo — runs
  out of the box and every drafted line traces back to something in the input.
- ``default_llm()`` returns the real client when ``ANTHROPIC_API_KEY`` is set
  and the mock otherwise, so identical code runs offline in CI and for real in
  production.

The mock is intentionally faithful: it never invents a problem, a medication, or
a warning that is not in the input. That way the offline path exercises the
honest case, and the grounding checks have real text to trace against.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMResult:
    """One model reply: free text and/or a forced-tool structured payload."""

    text: str = ""
    tool_input: Optional[dict] = None


class BaseLLM:
    """The one interface every surface depends on."""

    def complete(self, *, system: str = "", user: str = "",
                 tool: Optional[dict] = None, max_tokens: int = 1024) -> LLMResult:
        raise NotImplementedError


class AnthropicLLM(BaseLLM):
    """Real client over the anthropic SDK. Lazy so import never needs a key."""

    def __init__(self, model: str = "claude-sonnet-5"):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        return self._client

    def complete(self, *, system="", user="", tool=None, max_tokens=1024) -> LLMResult:
        kw = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if tool is not None:
            kw["tools"] = [tool]
            kw["tool_choice"] = {"type": "tool", "name": tool["name"]}
        r = self.client.messages.create(**kw)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        ti = next(
            (dict(b.input) for b in r.content if getattr(b, "type", "") == "tool_use"),
            None,
        )
        return LLMResult(text=text, tool_input=ti)


# ---------------------------------------------------------------------------
# Offline draft writers
# ---------------------------------------------------------------------------
#
# Everything below produces the *draft* the mock hands back. It is plain,
# deterministic text processing — no model, no network. The point is faithful
# behaviour: each drafted line is built from real input text so the grounding
# checks downstream can trace it.


def _numbered_snippets(user: str) -> list[tuple[str, str]]:
    """Recover ``<id>: <text>`` snippet lines the caller put in the prompt.

    Snippets are passed as ``S1: ...`` style lines. We keep only those, so the
    surrounding instruction text never leaks into a draft.
    """
    out: list[tuple[str, str]] = []
    for raw in user.splitlines():
        raw = raw.strip()
        m = re.match(r"^([A-Za-z]+\d+)\s*[:.\)]\s*(.+)$", raw)
        if m:
            out.append((m.group(1), m.group(2).strip()))
    return out


# Cue words that sort a chart snippet into a brief section. First match wins.
_PROBLEM_CUES = ("diagnos", "history of", "presents with", "admitted", "acute",
                 "chronic", "infection", "failure", "disease", "fracture",
                 "pain", "elevated", "abnormal", "positive for", "assessment",
                 "diabetes", "hypertension", "asthma", "copd", "cancer",
                 "stroke", "depression", "anxiety", "arthritis", "pneumonia")
_MED_CUES = ("mg", "dose", "tablet", "capsule", "inhaler", "insulin", "started on",
             "prescrib", "medication", "daily", "twice", "titrat", "stopped",
             "discontinu", "increased", "decreased", "units")
_FOLLOWUP_CUES = ("follow up", "follow-up", "return", "appointment", "recheck",
                  "refer", "schedule", "clinic in", "weeks", "next visit")


def _classify_snippet(text: str) -> str:
    low = text.lower()
    if any(c in low for c in _FOLLOWUP_CUES):
        return "Open items / follow-up"
    if any(c in low for c in _MED_CUES):
        return "Medications"
    if any(c in low for c in _PROBLEM_CUES):
        return "Active problems"
    return "Other context"


def _condense(text: str, limit: int = 160) -> str:
    """Trim a snippet to a short, clause-level brief line."""
    text = text.strip().rstrip(".")
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


def _mock_prebrief(user: str) -> dict:
    """Group chart snippets into a short, cited brief.

    Every point cites the snippet id it came from, so nothing in the brief is
    unattributed.
    """
    snippets = _numbered_snippets(user)
    order = ["Active problems", "Medications", "Open items / follow-up", "Other context"]
    buckets: dict[str, list[dict]] = {k: [] for k in order}
    for sid, text in snippets:
        section = _classify_snippet(text)
        buckets[section].append({"text": _condense(text), "source_ids": [sid]})

    sections = [{"heading": h, "points": buckets[h]} for h in order if buckets[h]]
    if not sections:
        sections = [{"heading": "Active problems",
                     "points": [{"text": "No chart snippets were provided.",
                                 "source_ids": []}]}]
    return {"sections": sections}


def _mock_discharge(user: str) -> dict:
    """Draft a discharge course + patient instructions from encounter snippets.

    Each drafted sentence is built from one snippet and cites it, so the
    grounding check has a real source to trace against.
    """
    snippets = _numbered_snippets(user)
    course, instructions = [], []
    for sid, text in snippets:
        low = text.lower()
        line = {"text": _condense(text, 200), "source_ids": [sid]}
        if any(c in low for c in _FOLLOWUP_CUES) or any(c in low for c in _MED_CUES) \
                or "call" in low or "return" in low or "if you" in low:
            instructions.append(line)
        else:
            course.append(line)
    if not course and instructions:
        course = instructions[:1]
    doc = {"sections": [
        {"heading": "Brief hospital course",
         "sentences": course or [{"text": "No encounter detail was provided.",
                                  "source_ids": []}]},
        {"heading": "Discharge instructions",
         "sentences": instructions or [{"text": "Follow up with your clinician as advised.",
                                        "source_ids": []}]},
    ]}
    return doc


# Medical term -> plain-English replacement. Longer phrases first so they win
# over any shorter fragment they contain.
_PLAIN_TERMS = [
    ("chronic obstructive pulmonary disease", "long-term lung disease"),
    ("myocardial infarction", "heart attack"),
    ("cerebrovascular accident", "stroke"),
    ("shortness of breath", "shortness of breath"),
    ("difficulty breathing", "trouble breathing"),
    ("blood pressure", "blood pressure"),
    ("emergency department", "emergency room"),
    ("as needed", "as needed"),
    ("bronchodilator", "inhaler medicine"),
    ("anticoagulant", "blood thinner"),
    ("hypertension", "high blood pressure"),
    ("hypotension", "low blood pressure"),
    ("exacerbation", "flare-up"),
    ("necessitating", "so you need"),
    ("administered", "given"),
    ("administer", "take"),
    ("discontinue", "stop"),
    ("analgesia", "pain relief"),
    ("analgesic", "pain medicine"),
    ("antibiotics", "antibiotics"),
    ("cellulitis", "skin infection"),
    ("laceration", "cut"),
    ("contusion", "bruise"),
    ("hemorrhage", "bleeding"),
    ("haemorrhage", "bleeding"),
    ("syncope", "fainting"),
    ("dyspnea", "shortness of breath"),
    ("dyspnoea", "shortness of breath"),
    ("pyrexia", "fever"),
    ("febrile", "feverish"),
    ("emesis", "vomiting"),
    ("nausea", "feeling sick"),
    ("erythema", "redness"),
    ("edema", "swelling"),
    ("oedema", "swelling"),
    ("myalgia", "muscle pain"),
    ("sutures", "stitches"),
    ("ambulate", "walk"),
    ("orally", "by mouth"),
    ("renal", "kidney"),
    ("hepatic", "liver"),
    ("commence", "start"),
    ("utilize", "use"),
    ("prior to", "before"),
    ("in the event that", "if"),
    ("approximately", "about"),
    ("sufficient", "enough"),
    ("twice daily", "two times a day"),
    ("once daily", "one time a day"),
    ("immediately", "right away"),
]

# Words that mark a sentence as safety-critical; the simplifier must never drop
# one of these sentences.
_SAFETY_MARKERS = ("emergency", "911", "call your", "call the", "seek", "immediately",
                   "right away", "do not", "don't", "stop taking", "worsen", "worse",
                   "chest pain", "trouble breathing", "difficulty breathing",
                   "bleeding", "allergic", "warning", "danger", "return to",
                   "follow up", "follow-up", "if you develop", "if you have")


def is_safety_sentence(sentence: str) -> bool:
    low = sentence.lower()
    return any(m in low for m in _SAFETY_MARKERS)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on end punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _plainen(sentence: str) -> str:
    """Swap jargon for plain words (case-insensitive, phrase-aware)."""
    out = sentence
    for term, plain in _PLAIN_TERMS:
        if term == plain:
            continue
        out = re.sub(re.escape(term), plain, out, flags=re.IGNORECASE)
    return out


def _shorten(sentence: str) -> list[str]:
    """Break one long sentence into shorter ones at natural seams.

    Shorter sentences and plainer words are what actually pull the reading grade
    down, so this is where the simplification happens.
    """
    seams = [", and ", "; ", ", which ", ", but ", ", so ", " because "]
    pieces = [sentence]
    for seam in seams:
        nxt: list[str] = []
        for p in pieces:
            nxt.extend(re.split(seam, p))
        pieces = nxt
    cleaned = []
    for p in pieces:
        p = p.strip().strip(",;")
        if not p:
            continue
        p = p[0].upper() + p[1:]
        if not p.endswith((".", "!", "?")):
            p += "."
        cleaned.append(p)
    return cleaned or [sentence]


def _mock_simplify(text: str) -> str:
    """Rewrite clinical text into plainer, shorter sentences.

    Safety-critical sentences are simplified word-for-word but never split away
    or dropped, so a warning cannot get lost in the rewrite.
    """
    out_sentences: list[str] = []
    for sent in split_sentences(text):
        plain = _plainen(sent)
        if is_safety_sentence(sent):
            # Keep the warning whole; only swap jargon.
            if not plain.endswith((".", "!", "?")):
                plain += "."
            out_sentences.append(plain[0].upper() + plain[1:])
        else:
            out_sentences.extend(_shorten(plain))
    return " ".join(out_sentences).strip()


class MockLLM(BaseLLM):
    """Deterministic offline stand-in. No key, no network, stable output."""

    def complete(self, *, system="", user="", tool=None, max_tokens=1024) -> LLMResult:
        name = tool.get("name") if tool else None
        if name == "write_prebrief":
            return LLMResult(tool_input=_mock_prebrief(user))
        if name == "write_discharge":
            return LLMResult(tool_input=_mock_discharge(user))
        if name == "simplify_text":
            return LLMResult(tool_input={"simplified": _mock_simplify(user)})
        if tool is not None:
            return LLMResult(tool_input={})
        return LLMResult(text=_mock_simplify(user))


def default_llm() -> BaseLLM:
    """Real client when a key is present, else the offline mock."""
    return AnthropicLLM() if os.environ.get("ANTHROPIC_API_KEY") else MockLLM()
