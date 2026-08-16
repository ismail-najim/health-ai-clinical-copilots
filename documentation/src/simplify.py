"""Plain-English rewrite — make medical text a patient can actually read.

Given clinical text, this rewrites it into plainer, shorter sentences aimed at a
patient reading level. The catch that matters: **simpler must never mean "lost
the warning."** So on top of the rewrite there is a safety check that finds every
safety-critical instruction in the source — danger signs, when to seek emergency
care, medication dose and timing, follow-up — and confirms each one survives in
the plain version. If a warning is dropped, the rewrite is flagged and blocked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .eval import DEFAULT_SUPPORT_THRESHOLD, reading_level_change
from .llm import BaseLLM, default_llm, is_safety_sentence, split_sentences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SIMPLIFY_TOOL = {
    "name": "simplify_text",
    "description": "Rewrite clinical text into plain, patient-friendly language.",
    "input_schema": {
        "type": "object",
        "properties": {"simplified": {"type": "string"}},
        "required": ["simplified"],
    },
}

SIMPLIFY_SYSTEM = (
    "Rewrite this clinical text for a patient at about a 6th-to-8th grade reading "
    "level. Keep it accurate. Use short sentences and plain words, and define any "
    "medical term you must keep. You MUST preserve every safety-critical "
    "instruction exactly in meaning: warning signs, when to seek emergency care, "
    "medication names, doses and timing, and follow-up appointments. Never drop, "
    "soften, or merge a warning."
)


@dataclass
class SafetyReport:
    """Did every safety-critical instruction in the source survive the rewrite?"""

    source_warnings: list[str] = field(default_factory=list)
    preserved: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)

    @property
    def preservation_rate(self) -> float:
        if not self.source_warnings:
            return 1.0
        return round(len(self.preserved) / len(self.source_warnings), 4)

    @property
    def blocked(self) -> bool:
        """True when a warning was lost — the rewrite must not be released."""
        return len(self.dropped) > 0


@dataclass
class SimplifyResult:
    source: str
    simplified: str
    reading: dict
    safety: SafetyReport

    @property
    def got_simpler(self) -> bool:
        return bool(self.reading.get("got_simpler"))

    @property
    def ok(self) -> bool:
        return self.got_simpler and not self.safety.blocked


def extract_warnings(text: str) -> list[str]:
    """Pull the safety-critical sentences out of the source text."""
    return [s for s in split_sentences(text) if is_safety_sentence(s)]


def safety_check(source: str, simplified: str,
                 threshold: float = DEFAULT_SUPPORT_THRESHOLD) -> SafetyReport:
    """Confirm each safety-critical instruction survives the rewrite.

    Every warning sentence in the source is matched against the simplified text
    by TF-IDF cosine similarity. A warning that has no close match in the plain
    version counts as dropped — tracked separately so an over-simplification that
    loses a danger sign is caught, whatever the reading score.
    """
    warnings = extract_warnings(source)
    if not warnings:
        return SafetyReport()
    simp_sentences = split_sentences(simplified) or [simplified]
    vect = TfidfVectorizer().fit(warnings + simp_sentences)
    simp_mat = vect.transform(simp_sentences)
    preserved, dropped = [], []
    for w in warnings:
        sim = cosine_similarity(vect.transform([w]), simp_mat)[0]
        (preserved if float(sim.max()) >= threshold else dropped).append(w)
    return SafetyReport(source_warnings=warnings, preserved=preserved, dropped=dropped)


def simplify(text: str, llm: Optional[BaseLLM] = None,
             threshold: float = DEFAULT_SUPPORT_THRESHOLD) -> SimplifyResult:
    """Rewrite ``text`` into plain language and check nothing important was lost.

    Returns the plain version, the reading-level change, and the safety report.
    """
    llm = llm or default_llm()
    text = (text or "").strip()

    result = llm.complete(system=SIMPLIFY_SYSTEM, user=text, tool=SIMPLIFY_TOOL,
                          max_tokens=900)
    simplified = ""
    if result.tool_input:
        simplified = str(result.tool_input.get("simplified", "")).strip()
    if not simplified:
        simplified = result.text.strip()

    reading = reading_level_change(text, simplified)
    safety = safety_check(text, simplified, threshold=threshold)
    return SimplifyResult(source=text, simplified=simplified, reading=reading, safety=safety)
