"""The main decision-support flow.

    retrieve  ->  ask the model to answer USING ONLY the retrieved snippets, with
    an inline [id] citation on every sentence  ->  a deterministic CITATION GATE
    that deletes any sentence lacking a real, in-context citation  ->  if nothing
    survives, abstain with an honest "not enough evidence to answer".

The gate is the point. A sentence the model cannot tie to a retrieved source is
removed, not softened. This is the "no source, no answer" rule, enforced in code
rather than trusted to the prompt. The clinician sees only lines that carry a
verifiable citation, plus the list of any lines that were removed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .corpus import corpus_as_dicts
from .llm import BaseLLM, default_llm
from .retriever import Retriever

ABSTAIN_MESSAGE = "Not enough evidence to answer from the available guidelines."

SYSTEM_PROMPT = """You are a decision-support assistant for licensed clinicians.
Answer the clinical question USING ONLY the numbered CONTEXT snippets below.

Rules:
- End EVERY clinical sentence with the citation id(s) it comes from, e.g. "... is
  first-line [G1]."
- Use only the [id]s that appear in CONTEXT. Never cite anything not shown.
- Present guideline-concordant options for a clinician to weigh. Do not give a
  patient-specific directive or a specific dose command; the clinician decides.
- If the CONTEXT does not support an answer, reply with exactly: INSUFFICIENT_EVIDENCE
"""

_CITE = re.compile(r"\[([A-Za-z0-9]+)\]")
# Short connective/framing lines that are allowed through without a citation.
_FRAMING = re.compile(
    r"^(options|consider|note|summary|in summary|the clinician|this is not|"
    r"decision support)\b", re.I)


@dataclass
class CDSResult:
    question: str
    answer: str
    abstained: bool
    kept: list[str] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)


def _is_clinical(sentence: str) -> bool:
    """A sentence making a factual clinical assertion must be cited. Short
    framing lines (headers, 'Options include:', disclaimers) are exempt."""
    s = sentence.strip()
    if not s or s.endswith(":"):
        return False
    if _FRAMING.match(s):
        return False
    return len(s.split()) >= 4


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def citation_gate(answer_text: str, valid_ids: set[str]) -> tuple[list[str], list[dict]]:
    """Keep only sentences whose citations are all present in valid_ids.

    Returns (kept_sentences, dropped_records). A clinical sentence with no
    citation, or citing an id not in context, is dropped. Non-clinical framing
    lines pass through untouched.
    """
    kept: list[str] = []
    dropped: list[dict] = []
    for sentence in _split_sentences(answer_text):
        cited = set(_CITE.findall(sentence))
        if not _is_clinical(sentence):
            kept.append(sentence)
            continue
        if cited and cited.issubset(valid_ids):
            kept.append(sentence)
        else:
            reason = "no citation" if not cited else "citation not in retrieved context"
            dropped.append({"sentence": sentence, "reason": reason})
    return kept, dropped


class GuidelineCDS:
    """Wires retrieval, generation, and the citation gate into one call."""

    def __init__(self, llm: Optional[BaseLLM] = None,
                 retriever: Optional[Retriever] = None, k: int = 3):
        self.llm = llm or default_llm()
        self.retriever = retriever or Retriever()
        self.k = k

    def _build_prompt(self, question: str, hits: list[dict]) -> str:
        lines = [f"[{h['id']}] {h['text']}" for h in hits]
        block = "\n".join(lines)
        return f"CONTEXT:\n{block}\n\nCLINICAL QUESTION: {question}"

    def answer(self, question: str) -> CDSResult:
        hits = self.retriever.retrieve(question, k=self.k)

        # No supporting snippet at all -> abstain before calling the model.
        if not hits:
            return CDSResult(question=question, answer=ABSTAIN_MESSAGE,
                             abstained=True, retrieved=[])

        prompt = self._build_prompt(question, hits)
        raw = self.llm.complete(system=SYSTEM_PROMPT, user=prompt).text.strip()

        # The model itself may signal insufficient evidence.
        if not raw or raw.upper().startswith("INSUFFICIENT_EVIDENCE"):
            return CDSResult(question=question, answer=ABSTAIN_MESSAGE,
                             abstained=True, retrieved=hits)

        valid_ids = {h["id"] for h in hits}
        kept, dropped = citation_gate(raw, valid_ids)

        # Nothing with a real citation survived -> abstain honestly.
        survivors = [s for s in kept if _is_clinical(s)]
        if not survivors:
            return CDSResult(question=question, answer=ABSTAIN_MESSAGE,
                             abstained=True, kept=[], dropped=dropped,
                             retrieved=hits)

        # Sources actually cited by the surviving text.
        cited_ids = set()
        for s in kept:
            cited_ids.update(_CITE.findall(s))
        sources = [h for h in hits if h["id"] in cited_ids]

        return CDSResult(
            question=question,
            answer=" ".join(kept),
            abstained=False,
            kept=kept,
            dropped=dropped,
            sources=sources,
            retrieved=hits,
        )
