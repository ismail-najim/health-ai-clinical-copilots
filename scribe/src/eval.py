"""Two numbers, tracked separately, because they fail differently.

- **Grounding rate** — of the sentences the note wrote, how many are actually
  backed by the transcript. Low grounding means the scribe made things up
  (hallucination). This reads straight off the confabulation check.

- **Omission rate** — of the facts a reference note captured, how many the scribe
  dropped. Low grounding and high omission are different failures: a note can be
  perfectly grounded and still be unsafe because it left out an allergy or a med
  change. Reporting one number hides the other, so we keep them apart.

Both are computed offline with TF-IDF cosine, no network and no API key.
"""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .scribe import Note, _split_sentences


@dataclass
class GroundingReport:
    grounding_rate: float
    supported: int
    total: int
    unsupported_sentences: list[str]


@dataclass
class OmissionReport:
    omission_rate: float
    missed: list[str]
    reference_facts: int
    covered: int


def grounding_rate(note: Note) -> GroundingReport:
    """Fraction of note sentences the confabulation check marked supported."""
    total = len(note.sentences)
    supported = sum(1 for s in note.sentences if s.status == "supported")
    missed = [s.text for s in note.sentences if s.status == "unsupported"]
    rate = supported / total if total else 0.0
    return GroundingReport(grounding_rate=round(rate, 4), supported=supported,
                           total=total, unsupported_sentences=missed)


def omission_rate(note: Note, reference_note: str, *,
                  coverage_threshold: float = 0.18) -> OmissionReport:
    """How many reference-note facts the generated note failed to cover.

    Each reference-note sentence is treated as one clinically-important fact.
    A fact counts as covered if some generated-note sentence is similar enough
    to it; otherwise it is an omission.
    """
    reference_facts = _split_sentences(reference_note)
    generated = [s.text for s in note.sentences]
    if not reference_facts:
        return OmissionReport(0.0, [], 0, 0)
    if not generated:
        return OmissionReport(1.0, list(reference_facts), len(reference_facts), 0)

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(reference_facts + generated)
    ref_vecs = matrix[: len(reference_facts)]
    gen_vecs = matrix[len(reference_facts):]
    sims = cosine_similarity(ref_vecs, gen_vecs)  # (n_reference, n_generated)

    missed = []
    covered = 0
    for fact, row in zip(reference_facts, sims):
        best = float(row.max()) if row.size else 0.0
        if best >= coverage_threshold:
            covered += 1
        else:
            missed.append(fact)
    rate = len(missed) / len(reference_facts)
    return OmissionReport(omission_rate=round(rate, 4), missed=missed,
                          reference_facts=len(reference_facts), covered=covered)
