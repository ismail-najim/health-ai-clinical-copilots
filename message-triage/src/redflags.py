"""Deterministic emergency gate — the first thing every message hits.

This is a plain pattern layer, not a model. It scans the raw patient message
for the signs of a medical emergency (chest pain, trouble breathing, stroke
signs, suicidal thoughts, severe bleeding, and so on). If it finds one, the
pipeline stops here and returns an urgent "seek emergency care" response. The
message never reaches the urgency classifier or the reply drafter.

Two design choices matter:

1. It sits *above* the model. No prompt, tone, or model output can switch it
   off. An emergency phrase always wins.
2. It is tuned to over-warn. Missing an emergency is the worst error this tool
   can make, so a few false alarms are an acceptable price. When in doubt, the
   gate fires.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


# (regex, category) — matched case-insensitively against the message.
# Patterns lean broad and include the idioms patients actually type, because
# recall on real emergencies is the whole point.
EMERGENCY_PATTERNS: List[Tuple[str, str]] = [
    (r"\bchest (pain|pressure|tight(ness)?|heavi(ness|er)|discomfort)\b"
     r"|elephant.{0,20}chest|chest.{0,20}(crush|squeez)", "possible cardiac event"),
    (r"\b(can'?t|cannot|couldn'?t|hard to|struggling to|trouble|difficulty)"
     r"\s+breath(e|ing)?\b|gasping|short(ness)? of breath|can'?t catch my breath"
     r"|turning blue", "breathing difficulty"),
    (r"\b(face|arm|leg|one side).{0,20}(droop|numb|weak|paraly)"
     r"|slurr(ed|ing)?\s+speech|can'?t speak|sudden(ly)?\s+confus", "possible stroke"),
    (r"\b(suicid|kill myself|end my life|end it all|take my (own )?life"
     r"|harm myself|hurt myself|don'?t want to (live|be here|wake up)"
     r"|better off dead|no reason to live)\b", "self-harm risk"),
    (r"\b(severe|heavy|profuse|gushing|won'?t stop|can'?t stop)\s+bleed(ing)?\b"
     r"|cough(ed|ing)?\s+up blood|vomit(ed|ing)?\s+blood"
     r"|blood.{0,10}won'?t stop", "severe bleeding"),
    (r"\bthroat (closing|swelling|tight)|anaphyla|lips?.{0,10}swell"
     r"|tongue.{0,10}swell|can'?t swallow|whole body.{0,10}(hives|rash)"
     r"\b", "possible anaphylaxis"),
    (r"\bworst headache (of my life|ever)\b|sudden.{0,15}severe headache"
     r"|thunderclap headache", "sudden severe headache"),
    (r"\b(passed out|fainted|unconscious|unresponsive|won'?t wake up"
     r"|seizure|convuls|overdos(e|ed))\b", "loss of consciousness or seizure"),
]

# Shown to the patient when the gate fires. Neutral, urgent, and it does not
# guess a diagnosis — it points to emergency care and to the on-call team.
EMERGENCY_GUIDANCE = (
    "This message may indicate a medical emergency. If this is an emergency, "
    "call your local emergency number now (for example 911 in the US or 112 in "
    "the EU) or go to the nearest emergency department. This message is being "
    "routed to the on-call clinician right away."
)


@dataclass
class RedFlagResult:
    """Outcome of the emergency screen for one message."""

    is_emergency: bool
    category: Optional[str] = None
    matched_text: Optional[str] = None
    guidance: str = ""

    @property
    def blocks_routine_reply(self) -> bool:
        """True when the pipeline must not draft a routine reply."""
        return self.is_emergency


def screen(message: str) -> RedFlagResult:
    """Check a message for emergency signs. Runs before any model call.

    Returns a :class:`RedFlagResult`. When ``is_emergency`` is true the caller
    must surface :attr:`RedFlagResult.guidance` and must not draft a routine
    reply.
    """
    text = message or ""
    for pattern, category in EMERGENCY_PATTERNS:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return RedFlagResult(
                is_emergency=True,
                category=category,
                matched_text=m.group(0),
                guidance=EMERGENCY_GUIDANCE,
            )
    return RedFlagResult(is_emergency=False)
