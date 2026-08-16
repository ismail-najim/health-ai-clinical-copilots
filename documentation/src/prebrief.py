"""Pre-visit brief — a short, cited summary of a patient's notes.

Given a set of note snippets (each with an id), this drafts a one-screen brief:
the active problems, current medications, and open follow-up items, grouped into
sections. **Every point cites the snippet id it came from**, so a clinician can
click straight back to the source line. A point with no citation is surfaced, not
hidden — the whole promise is that nothing in the brief is unattributed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .llm import BaseLLM, default_llm

BRIEF_TOOL = {
    "name": "write_prebrief",
    "description": "Group chart snippets into a short, sectioned, cited pre-visit brief.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "points": {
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
                    "required": ["heading", "points"],
                },
            },
        },
        "required": ["sections"],
    },
}

BRIEF_SYSTEM = (
    "You write a one-screen pre-visit brief for a clinician from a set of chart "
    "note snippets. Group the key points into sections: Active problems, "
    "Medications, and Open items / follow-up. Keep each point to a short clause. "
    "EVERY point must cite the snippet id(s) that support it in source_ids. Do not "
    "introduce a problem, medication, or follow-up that is not in the snippets."
)


@dataclass
class BriefPoint:
    text: str
    source_ids: list[str] = field(default_factory=list)

    @property
    def cited(self) -> bool:
        return len(self.source_ids) > 0


@dataclass
class BriefSection:
    heading: str
    points: list[BriefPoint] = field(default_factory=list)


@dataclass
class Brief:
    sections: list[BriefSection] = field(default_factory=list)

    @property
    def points(self) -> list[BriefPoint]:
        return [p for sec in self.sections for p in sec.points]

    @property
    def citation_rate(self) -> float:
        """Fraction of points that carry at least one source id."""
        pts = self.points
        if not pts:
            return 1.0
        return round(sum(1 for p in pts if p.cited) / len(pts), 4)

    @property
    def uncited(self) -> list[str]:
        return [p.text for p in self.points if not p.cited]

    def render(self) -> str:
        lines = []
        for sec in self.sections:
            lines.append(f"## {sec.heading}")
            for p in sec.points:
                cite = f"  [{', '.join(p.source_ids)}]" if p.source_ids else "  [uncited]"
                lines.append(f"- {p.text}{cite}")
            lines.append("")
        return "\n".join(lines).strip()


def _normalize(snippets) -> list[dict]:
    """Accept snippets as plain strings or as ``{'id','text'}`` dicts."""
    out = []
    for i, s in enumerate(snippets):
        if isinstance(s, dict):
            sid = str(s.get("id") or s.get("source_id") or f"S{i + 1}")
            text = str(s.get("text", "")).strip()
        else:
            sid, text = f"S{i + 1}", str(s).strip()
        if text:
            out.append({"id": sid, "text": text})
    return out


def generate_brief(snippets, llm: Optional[BaseLLM] = None) -> Brief:
    """Draft a cited pre-visit brief from note snippets.

    ``snippets`` may be strings or ``{'id', 'text'}`` dicts. Each drafted point
    is validated so its ``source_ids`` only reference real snippet ids.
    """
    llm = llm or default_llm()
    items = _normalize(snippets)
    valid_ids = {it["id"] for it in items}
    user = "\n".join(f'{it["id"]}: {it["text"]}' for it in items)

    result = llm.complete(system=BRIEF_SYSTEM, user=user, tool=BRIEF_TOOL, max_tokens=900)
    payload = result.tool_input or {"sections": []}

    sections = []
    for sec in payload.get("sections", []):
        points = []
        for p in sec.get("points", []):
            ids = [sid for sid in p.get("source_ids", []) if sid in valid_ids]
            points.append(BriefPoint(text=str(p.get("text", "")).strip(), source_ids=ids))
        points = [p for p in points if p.text]
        if points:
            sections.append(BriefSection(heading=str(sec.get("heading", "")).strip(),
                                         points=points))
    return Brief(sections=sections)
