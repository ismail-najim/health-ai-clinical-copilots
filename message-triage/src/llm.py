"""Language-model client with a deterministic offline stand-in.

Everything in this tool talks to a model through one tiny interface,
``BaseLLM.complete(...)``. Two implementations ship here:

- ``AnthropicLLM`` wraps the official ``anthropic`` SDK. It is lazy: importing
  this module never opens a network connection or needs a key. The client is
  built the first time you actually call it.
- ``MockLLM`` is a deterministic stand-in. It classifies urgency, condenses a
  question, and writes a plausible reply with no key and no network, so tests
  and the offline demo run out of the box and always give the same answer.

``default_llm()`` returns the real client when ``ANTHROPIC_API_KEY`` is set and
the mock otherwise. The same pipeline therefore runs offline in CI and for real
in production without a code change.
"""
from __future__ import annotations

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
    """The one interface the rest of the tool depends on."""

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


# --- offline stand-in helpers ----------------------------------------------

def _field(user: str, label: str) -> str:
    """Pull one labelled block out of a prompt.

    Callers format prompts as ``LABEL:\\n<value>`` blocks separated by blank
    lines. This lets the mock recover the patient message, the condensed
    question, and the record without guessing.
    """
    pattern = rf"{re.escape(label)}:\s*\n?(.*?)(?:\n[A-Z][A-Z ]+:|\Z)"
    m = re.search(pattern, user, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


# Words that push a message up the urgency scale. These are for the offline
# stand-in only; the real classifier is trained in triage.py, and the
# deterministic emergency gate in redflags.py sits above both of them.
_URGENT_CUES = (
    "fever", "days", "worse", "worsening", "spreading", "swelling", "swollen",
    "vomit", "can't keep", "cannot keep", "dizzy", "dizziness", "infection",
    "pus", "blood in", "dehydrated", "no better", "getting worse", "high temp",
    "won't go away", "wont go away", "rash", "urgent", "asap", "today",
)
_ROUTINE_CUES = (
    "refill", "prescription", "appointment", "results", "form", "letter",
    "renew", "schedule", "reschedule", "question about", "wondering",
    "follow up", "follow-up", "vaccine", "record",
)


def _mock_urgency(user: str) -> dict:
    """Heuristic three-way urgency for the offline path."""
    message = _field(user, "PATIENT MESSAGE") or user
    low = message.lower()
    if any(c in low for c in _URGENT_CUES):
        return {
            "urgency": "urgent",
            "reason": "Message mentions a symptom that may be worsening or "
                      "time-sensitive; a clinician should review it soon.",
            "needs_clinician": True,
        }
    if any(c in low for c in _ROUTINE_CUES):
        return {
            "urgency": "routine",
            "reason": "Message reads as an administrative or general question "
                      "with no time pressure.",
            "needs_clinician": True,
        }
    return {
        "urgency": "routine",
        "reason": "No time-sensitive symptoms detected; treat as a routine ask.",
        "needs_clinician": True,
    }


def _mock_summary(user: str) -> str:
    """Condense the message to a single question for the offline path."""
    message = _field(user, "PATIENT MESSAGE") or user
    # Prefer a sentence the patient framed as a question.
    sentences = re.split(r"(?<=[.?!])\s+", message.strip())
    for s in sentences:
        if "?" in s:
            return s.strip()
    # Otherwise use the last non-empty sentence, phrased as an ask.
    tail = next((s.strip() for s in reversed(sentences) if s.strip()), message.strip())
    tail = tail.rstrip(".!")
    if not tail:
        return "What would the patient like to know?"
    return f"The patient is asking: {tail[0].lower()}{tail[1:]}?"


def _mock_reply(user: str) -> str:
    """Write a warm, grounded draft for the offline path.

    The draft answers the condensed question, works in any record facts the
    caller supplied, and closes by pointing back to the care team. It never
    diagnoses and never tells the patient a symptom is nothing to worry about.
    """
    question = _field(user, "SUMMARIZED QUESTION")
    record = _field(user, "RECORD")
    lines = ["Thanks for reaching out, and I'm sorry you've been dealing with this."]
    if question:
        ask = question.rstrip("?").strip()
        lines.append(f"You asked about {ask.lower()}.")
    if record:
        first_fact = record.splitlines()[0].strip("-* ").strip()
        if first_fact:
            lines.append(f"Looking at your chart, I can see {first_fact.lower()}.")
    lines.append(
        "Here is some general guidance to consider, though your care team will "
        "confirm what's right for you before anything is finalized."
    )
    lines.append(
        "If your symptoms get worse, don't ease up, or you feel this can't wait, "
        "please contact the clinic or seek urgent care rather than waiting for a reply."
    )
    lines.append("A member of your care team will review this and follow up with you.")
    return "\n\n".join(lines)


class MockLLM(BaseLLM):
    """Deterministic offline stand-in. No key, no network, stable output."""

    def complete(self, *, system="", user="", tool=None, max_tokens=1024) -> LLMResult:
        if tool is not None:
            if tool.get("name") == "classify_urgency":
                return LLMResult(tool_input=_mock_urgency(user))
            return LLMResult(tool_input={})
        low_sys = system.lower()
        if "condense" in low_sys or "single question" in low_sys:
            return LLMResult(text=_mock_summary(user))
        return LLMResult(text=_mock_reply(user))


def default_llm() -> BaseLLM:
    """Real client when a key is present, else the deterministic offline mock."""
    return AnthropicLLM() if os.environ.get("ANTHROPIC_API_KEY") else MockLLM()
