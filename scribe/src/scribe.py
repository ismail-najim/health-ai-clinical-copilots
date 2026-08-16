"""Transcript to grounded clinical note.

Three stages, in order:

1. **Note generation.** Hand the line-numbered transcript to the model and ask
   for a sectioned note through a forced tool. Offline this is the deterministic
   mock; with a key it is the real client.
2. **Linked evidence.** For every note sentence, find the transcript line(s) it
   came from using offline TF-IDF cosine similarity. No embeddings, no network.
   Each sentence carries the lines that support it and how strong the match is.
3. **Confabulation check.** Any sentence whose best supporting line is too weak
   is flagged ``unsupported`` — the model wrote something the transcript does not
   back up. This is the honest guardrail: it surfaces made-up content instead of
   smoothing over it, so a clinician can catch it before signing.

Nothing here files a note. The output is a draft for a human to review and sign.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .llm import BaseLLM, default_llm

# The model must return a note through this tool, so the shape is guaranteed.
WRITE_NOTE_TOOL = {
    "name": "write_note",
    "description": "Write a structured clinical note from a visit transcript. "
                   "Use only what the transcript states; do not invent findings.",
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
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["heading", "sentences"],
                },
            }
        },
        "required": ["sections"],
    },
}

SYSTEM = (
    "You are a clinical scribe. Turn the numbered visit transcript into a concise, "
    "structured note. Every sentence must be supported by the transcript. Do not add "
    "findings, vitals, medications, or history that were not stated. Prefer omitting "
    "over inventing."
)


@dataclass
class TranscriptLine:
    line: int
    speaker: str
    text: str


@dataclass
class Evidence:
    """One supporting transcript line for a note sentence."""

    line: int
    text: str
    similarity: float


@dataclass
class NoteSentence:
    text: str
    heading: str
    evidence: list[Evidence] = field(default_factory=list)
    status: str = "supported"  # "supported" | "unsupported"

    @property
    def source_lines(self) -> list[int]:
        return [e.line for e in self.evidence]

    @property
    def best_similarity(self) -> float:
        return max((e.similarity for e in self.evidence), default=0.0)


@dataclass
class Note:
    sentences: list[NoteSentence] = field(default_factory=list)

    def by_section(self) -> list[tuple[str, list[NoteSentence]]]:
        out: list[tuple[str, list[NoteSentence]]] = []
        for s in self.sentences:
            if not out or out[-1][0] != s.heading:
                out.append((s.heading, []))
            out[-1][1].append(s)
        return out

    @property
    def flagged(self) -> list[NoteSentence]:
        return [s for s in self.sentences if s.status == "unsupported"]


_SPEAKER_RE = re.compile(r"^\s*(dr\.?|doctor|patient|pt|clinician|nurse)\s*[:\-]\s*",
                         re.IGNORECASE)


def parse_transcript(raw: str) -> list[TranscriptLine]:
    """Split a raw transcript into line-indexed, speaker-tagged utterances.

    Accepts ``Doctor: ...`` / ``Patient: ...`` style lines; a line with no tag
    keeps an empty speaker. Blank lines are dropped.
    """
    lines: list[TranscriptLine] = []
    for raw_line in raw.splitlines():
        text = raw_line.strip()
        if not text:
            continue
        m = _SPEAKER_RE.match(text)
        speaker = m.group(1).rstrip(":-. ").title() if m else ""
        body = _SPEAKER_RE.sub("", text).strip()
        if body:
            lines.append(TranscriptLine(line=len(lines), speaker=speaker, text=body))
    return lines


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class Scribe:
    """Runs the transcript through generation, linking, and the confab check."""

    def __init__(self, llm: Optional[BaseLLM] = None, *,
                 support_threshold: float = 0.18, max_evidence: int = 3):
        self.llm = llm or default_llm()
        # A sentence whose best transcript match scores below this is flagged.
        self.support_threshold = support_threshold
        self.max_evidence = max_evidence

    # -- stage 1 -----------------------------------------------------------
    def generate_note(self, transcript: list[TranscriptLine]) -> list[tuple[str, str]]:
        """Ask the model for a sectioned note. Returns (heading, sentence) pairs."""
        numbered = "\n".join(f"{ln.line}: "
                             f"{(ln.speaker + ': ') if ln.speaker else ''}{ln.text}"
                             for ln in transcript)
        user = ("Write the visit note for this transcript.\n\nTranscript:\n" + numbered)
        result = self.llm.complete(system=SYSTEM, user=user, tool=WRITE_NOTE_TOOL)
        data = result.tool_input or {}
        pairs: list[tuple[str, str]] = []
        for section in data.get("sections", []):
            heading = section.get("heading", "Note")
            for sentence in section.get("sentences", []):
                for one in _split_sentences(sentence):
                    pairs.append((heading, one))
        return pairs

    # -- stages 2 & 3 ------------------------------------------------------
    def link_and_check(self, note_pairs: list[tuple[str, str]],
                       transcript: list[TranscriptLine]) -> Note:
        """Attach supporting transcript lines to each sentence and flag the weak ones."""
        note = Note()
        if not note_pairs:
            return note
        if not transcript:
            for heading, text in note_pairs:
                note.sentences.append(
                    NoteSentence(text=text, heading=heading, status="unsupported"))
            return note

        line_texts = [ln.text for ln in transcript]
        sentence_texts = [t for _, t in note_pairs]
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(line_texts + sentence_texts)
        line_vecs = matrix[: len(line_texts)]
        sent_vecs = matrix[len(line_texts):]
        sims = cosine_similarity(sent_vecs, line_vecs)  # (n_sentences, n_lines)

        for (heading, text), row in zip(note_pairs, sims):
            ranked = sorted(range(len(row)), key=lambda i: row[i], reverse=True)
            evidence = []
            for idx in ranked[: self.max_evidence]:
                score = float(row[idx])
                if score <= 0.0:
                    break
                evidence.append(Evidence(line=transcript[idx].line,
                                         text=transcript[idx].text,
                                         similarity=round(score, 4)))
            sentence = NoteSentence(text=text, heading=heading, evidence=evidence)
            best = sentence.best_similarity
            sentence.status = ("supported" if best >= self.support_threshold
                               else "unsupported")
            # Keep only evidence at or above threshold on supported sentences;
            # on flagged ones keep the top near-miss so the reviewer sees why.
            if sentence.status == "supported":
                sentence.evidence = [e for e in evidence
                                     if e.similarity >= self.support_threshold] or evidence[:1]
            else:
                sentence.evidence = evidence[:1]
            note.sentences.append(sentence)
        return note

    def run(self, raw_transcript: str) -> tuple[Note, list[TranscriptLine]]:
        """Full pipeline: raw text in, grounded-and-checked note out."""
        transcript = parse_transcript(raw_transcript)
        pairs = self.generate_note(transcript)
        note = self.link_and_check(pairs, transcript)
        return note, transcript
