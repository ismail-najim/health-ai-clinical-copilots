"""Discharge summary — a hospital course + patient instructions, grounded.

Given an encounter summary (a set of snippets describing the stay), this drafts
two sections: a short **Brief hospital course** and plain **Discharge
instructions**. Then it runs a grounding check: every drafted sentence is
compared against the input, and any sentence the input does not support is
**flagged** rather than quietly filed. Unsupported sentences are the model
inventing detail — the check is what catches them before a clinician signs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .eval import DEFAULT_SUPPORT_THRESHOLD, support_scores
from .llm import BaseLLM, default_llm

DISCHARGE_TOOL = {
    "name": "write_discharge",
    "description": "Draft a brief hospital course and patient discharge instructions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "sentences": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "source_ids": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["text", "source_ids"],
                            },
                        },
                    },
                    "required": ["heading", "sentences"],
                },
            },
        },
        "required": ["sections"],
    },
}

DISCHARGE_SYSTEM = (
    "You draft a discharge summary from an encounter's notes. Produce two "
    "sections: 'Brief hospital course' and 'Discharge instructions'. The "
    "instructions must, when present in the notes, include medication changes "
    "with dose and timing, follow-up appointments, and warning signs / when to "
    "seek emergency care. EVERY sentence must cite the source snippet id(s) that "
    "support it. Do not invent findings, medications, or follow-ups."
)


@dataclass
class DischargeSentence:
    text: str
    source_ids: list[str] = field(default_factory=list)
    supported: bool = True
    support_score: float = 0.0


@dataclass
class DischargeSection:
    heading: str
    sentences: list[DischargeSentence] = field(default_factory=list)


@dataclass
class DischargeDoc:
    sections: list[DischargeSection] = field(default_factory=list)

    @property
    def all_sentences(self) -> list[DischargeSentence]:
        return [s for sec in self.sections for s in sec.sentences]

    @property
    def unsupported(self) -> list[DischargeSentence]:
        return [s for s in self.all_sentences if not s.supported]

    @property
    def has_unsupported(self) -> bool:
        return len(self.unsupported) > 0

    def render(self) -> str:
        lines = []
        for sec in self.sections:
            lines.append(f"## {sec.heading}")
            for s in sec.sentences:
                flag = "" if s.supported else "  ⚠ UNSUPPORTED — review"
                cite = f"  [{', '.join(s.source_ids)}]" if s.source_ids else ""
                lines.append(f"- {s.text}{cite}{flag}")
            lines.append("")
        return "\n".join(lines).strip()


def _normalize(encounter) -> list[dict]:
    """Accept the encounter as one text block, a list of strings, or dicts."""
    if isinstance(encounter, str):
        rows = [r.strip() for r in encounter.splitlines() if r.strip()]
        return [{"id": f"S{i + 1}", "text": r} for i, r in enumerate(rows)]
    out = []
    for i, s in enumerate(encounter):
        if isinstance(s, dict):
            sid = str(s.get("id") or s.get("source_id") or f"S{i + 1}")
            text = str(s.get("text", "")).strip()
        else:
            sid, text = f"S{i + 1}", str(s).strip()
        if text:
            out.append({"id": sid, "text": text})
    return out


def check_support(doc: DischargeDoc, sources: list[str], source_ids: list[str] | None = None,
                  threshold: float = DEFAULT_SUPPORT_THRESHOLD) -> DischargeDoc:
    """Flag any sentence the source snippets do not support.

    Fills in each sentence's ``supported`` flag and ``support_score`` from a
    TF-IDF cosine match against the sources. Standalone so it can be run over any
    draft, including one that was edited by hand.
    """
    sentences = [s.text for s in doc.all_sentences]
    scored = support_scores(sentences, sources, source_ids=source_ids, threshold=threshold)
    for sent_obj, result in zip(doc.all_sentences, scored):
        sent_obj.supported = result.supported
        sent_obj.support_score = result.score
    return doc


def generate_discharge(encounter, llm: Optional[BaseLLM] = None,
                       threshold: float = DEFAULT_SUPPORT_THRESHOLD) -> DischargeDoc:
    """Draft a discharge summary and flag any unsupported sentence.

    ``encounter`` may be one text block, a list of strings, or ``{'id','text'}``
    dicts. The returned doc has each sentence marked supported / unsupported
    against the input.
    """
    llm = llm or default_llm()
    items = _normalize(encounter)
    valid_ids = {it["id"] for it in items}
    sources = [it["text"] for it in items]
    source_ids = [it["id"] for it in items]
    user = "\n".join(f'{it["id"]}: {it["text"]}' for it in items)

    result = llm.complete(system=DISCHARGE_SYSTEM, user=user, tool=DISCHARGE_TOOL,
                          max_tokens=1200)
    payload = result.tool_input or {"sections": []}

    sections = []
    for sec in payload.get("sections", []):
        sents = []
        for s in sec.get("sentences", []):
            ids = [sid for sid in s.get("source_ids", []) if sid in valid_ids]
            text = str(s.get("text", "")).strip()
            if text:
                sents.append(DischargeSentence(text=text, source_ids=ids))
        if sents:
            sections.append(DischargeSection(heading=str(sec.get("heading", "")).strip(),
                                             sentences=sents))
    doc = DischargeDoc(sections=sections)
    return check_support(doc, sources, source_ids=source_ids, threshold=threshold)
