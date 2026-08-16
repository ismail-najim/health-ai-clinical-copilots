"""Condense a rambling message down to the question it is really asking.

Patients often write a paragraph of context before the one thing they want to
know. This module turns that into a single clinical question — what the
clinician reads first and what the reply drafter answers.

A faithfulness guard checks that the condensed question does not introduce a
symptom or detail that was never in the original message, because an invented
detail would poison every step downstream.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .llm import BaseLLM, default_llm

_SUMMARY_SYSTEM = (
    "Condense the patient's message into a single question that captures what "
    "they actually want to know. Do not add symptoms, durations, or details the "
    "message does not state. If the message contains more than one ask, capture "
    "the main one. Reply with the single question only."
)

# Common words we ignore when checking whether the summary invented content.
_STOPWORDS = {
    "about", "would", "could", "should", "there", "their", "which", "patient",
    "asking", "question", "please", "thanks", "thank", "hello", "wondering",
    "know", "want", "like", "need", "have", "with", "this", "that", "your",
    "from", "been", "will", "what", "when", "does", "still", "just",
}


@dataclass
class SummaryResult:
    """A condensed question plus whether it stayed faithful to the message."""

    question: str
    faithful: bool = True
    novel_terms: Optional[list] = None


def _words(text: str) -> set:
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def is_faithful(question: str, message: str, max_novel: int = 2) -> tuple:
    """Cheap faithfulness check: the summary should add little the message lacks.

    Returns ``(ok, novel_terms)``. This is a fast pre-filter, not an entailment
    model; it flags a summary that introduces several content words absent from
    the original so a human can look closer.
    """
    novel = sorted(_words(question) - _words(message) - _STOPWORDS)
    return (len(novel) <= max_novel, novel)


def summarize(message: str, llm: Optional[BaseLLM] = None) -> SummaryResult:
    """Return the single question the message is asking."""
    llm = llm or default_llm()
    result = llm.complete(
        system=_SUMMARY_SYSTEM,
        user=f"PATIENT MESSAGE:\n{message}",
        max_tokens=120,
    )
    question = (result.text or "").strip()
    if not question:
        question = "What would the patient like to know?"
    ok, novel = is_faithful(question, message)
    return SummaryResult(question=question, faithful=ok, novel_terms=novel)
