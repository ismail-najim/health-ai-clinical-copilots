"""Draft a reply for a clinician to approve — never send one automatically.

The drafter answers the condensed question in warm, plain language, works in
any facts from the patient's own record that were passed in, and always points
the patient back to the care team. It does not diagnose and it does not tell a
patient a symptom is nothing to worry about.

The approval gate is the important part. ``draft_reply`` returns a ``Draft``
with ``sent=False`` and ``approved=False``. The only way a draft becomes sent
is ``approve``, which a clinician calls after reviewing it. There is no path in
this module that sends on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from .llm import BaseLLM, default_llm

_REPLY_SYSTEM = (
    "You draft a reply to a patient's portal message for a clinician to review "
    "and send. You do not diagnose and you do not change medications. Answer the "
    "summarized question in warm, plain language, using the patient's record "
    "facts where they help. Never reassure a patient that a symptom is nothing "
    "to worry about, and if anything could be time-sensitive, tell them to "
    "contact the care team or seek urgent care. If a health misconception is "
    "present, gently correct it rather than agreeing with it. End by noting a "
    "clinician will review and follow up."
)


@dataclass
class Draft:
    """A reply draft and its approval state.

    A draft starts unapproved and unsent. It can only be sent through
    :func:`approve`, which records who signed off and when.
    """

    question: str
    body: str
    record: List[str] = field(default_factory=list)
    requires_approval: bool = True
    approved: bool = False
    sent: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None

    @property
    def status(self) -> str:
        if self.sent:
            return f"sent (approved by {self.approved_by})"
        return "awaiting clinician approval"


def draft_reply(question: str, message: str,
                record: Optional[List[str]] = None,
                llm: Optional[BaseLLM] = None) -> Draft:
    """Write a draft reply. The returned draft is never sent on its own."""
    llm = llm or default_llm()
    record = record or []
    record_block = "\n".join(f"- {fact}" for fact in record) if record else "(none)"
    user = (
        f"PATIENT MESSAGE:\n{message}\n\n"
        f"SUMMARIZED QUESTION:\n{question}\n\n"
        f"RECORD:\n{record_block}"
    )
    result = llm.complete(system=_REPLY_SYSTEM, user=user, max_tokens=700)
    body = (result.text or "").strip()
    return Draft(question=question, body=body, record=list(record))


def approve(draft: Draft, clinician: str) -> Draft:
    """The only send path. A clinician approves; the draft is marked sent.

    Raises ``ValueError`` if no clinician name is given, so an empty approval
    can never send a message.
    """
    if not clinician or not clinician.strip():
        raise ValueError("A clinician name is required to approve and send.")
    draft.approved = True
    draft.sent = True
    draft.approved_by = clinician.strip()
    draft.approved_at = datetime.now(timezone.utc).isoformat()
    return draft
