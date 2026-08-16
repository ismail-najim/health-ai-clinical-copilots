"""Language-model client with an injectable, deterministic offline stand-in.

The scribe calls a model through one small interface, `BaseLLM.complete(...)`.
Two implementations ship here:

- `AnthropicLLM` wraps the official `anthropic` SDK. It is lazy: importing this
  module never touches the network or requires a key. The client is only built
  the first time you actually call it.
- `MockLLM` is a deterministic stand-in that writes a plausible note from the
  transcript with no API key and no network. Every test and the offline demo
  path run on it, so the whole tool works out of the box.

`default_llm()` returns the real client when `ANTHROPIC_API_KEY` is set and the
mock otherwise, so the same code runs offline in CI and for real in production.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResult:
    """One model reply: free text and/or a forced-tool structured payload."""

    text: str = ""
    tool_input: Optional[dict] = None


class BaseLLM:
    """The one interface the scribe depends on."""

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


# --- offline note writer ----------------------------------------------------

# Section headings the mock groups utterances under, with the cue words that
# route a transcript line to a section. First match wins; anything unmatched
# lands in the History/Context bucket.
_SECTION_CUES = [
    ("Chief Complaint", ("here for", "come in", "bothering", "complain", "problem is",
                         "reason for", "brings you")),
    ("History of Present Illness", ("started", "since", "days", "weeks", "began", "onset",
                                    "pain", "ache", "cough", "fever", "worse", "better",
                                    "feel", "symptom", "hurts", "sore", "tired")),
    ("Medications and Allergies", ("taking", "medication", "allergic", "allergy", "dose",
                                   "mg", "prescribed", "pill", "ibuprofen", "penicillin",
                                   "insulin", "metformin")),
    ("Plan", ("recommend", "prescribe", "order", "refer", "follow up", "follow-up",
              "let's", "we'll", "start you", "x-ray", "test", "bloodwork", "come back")),
]


def _split_utterances(user: str) -> list[str]:
    """Recover the transcript lines the scribe passed in the prompt.

    The scribe sends lines as ``<i>: <text>``. We keep only those numbered lines
    so the surrounding instruction text ("Write the note...", "Transcript:") never
    leaks into the note. If no numbered lines are present we fall back to every
    non-empty line so the mock still produces something usable.
    """
    numbered = []
    for raw in user.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        m = re.match(r"^\d+\s*[:.\)]\s*(.+)$", raw)
        if m:
            numbered.append(m.group(1).strip())
    if numbered:
        return numbered
    return [raw.strip() for raw in user.splitlines() if raw.strip()]


def _strip_speaker(text: str) -> str:
    """Drop a leading ``Doctor:`` / ``Patient:`` speaker tag if present."""
    return re.sub(r"^\s*(dr\.?|doctor|patient|pt|clinician|nurse)\s*[:\-]\s*",
                  "", text, flags=re.IGNORECASE).strip()


def _mock_note(user: str) -> dict:
    """Write a plausible, transcript-grounded note with no network access.

    Every sentence is built from real utterance text, so the downstream
    evidence linker can trace each one back to the line it came from. This is
    deliberate: the offline path should behave like a faithful scribe, not
    invent content, so tests exercise the honest case by default.
    """
    utterances = _split_utterances(user)
    buckets: dict[str, list[str]] = {h: [] for h, _ in _SECTION_CUES}
    buckets["History and Context"] = []

    for utt in utterances:
        body = _strip_speaker(utt)
        if not body:
            continue
        low = body.lower()
        placed = False
        for heading, cues in _SECTION_CUES:
            if any(c in low for c in cues):
                buckets[heading].append(body)
                placed = True
                break
        if not placed:
            buckets["History and Context"].append(body)

    order = ["Chief Complaint", "History of Present Illness",
             "Medications and Allergies", "History and Context", "Plan"]
    sections = []
    for heading in order:
        items = buckets.get(heading, [])
        if not items:
            continue
        sentences = []
        for item in items:
            clause = item.rstrip(". ")
            # Turn an utterance into a flat clinical statement.
            sentences.append(f"{clause[0].upper()}{clause[1:]}." if clause else "")
        sentences = [s for s in sentences if s]
        if sentences:
            sections.append({"heading": heading, "sentences": sentences})

    if not sections:
        sections = [{"heading": "Visit Summary",
                     "sentences": ["No content was captured from the transcript."]}]
    return {"sections": sections}


class MockLLM(BaseLLM):
    """Deterministic offline stand-in. No key, no network, stable output."""

    def complete(self, *, system="", user="", tool=None, max_tokens=1024) -> LLMResult:
        if tool is not None and tool.get("name") == "write_note":
            return LLMResult(tool_input=_mock_note(user))
        # Generic structured fallback so an unknown forced tool still returns JSON.
        if tool is not None:
            return LLMResult(tool_input={})
        return LLMResult(text=json.dumps(_mock_note(user)))


def default_llm() -> BaseLLM:
    """Real client when a key is present, else the offline mock."""
    return AnthropicLLM() if os.environ.get("ANTHROPIC_API_KEY") else MockLLM()
