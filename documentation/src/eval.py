"""Metrics — kept as separate numbers because they fail in different ways.

Three things are measured here, and they are never collapsed into one score:

- **Reading level** (via ``textstat`` — Flesch-Kincaid grade, reading ease, SMOG)
  before and after a rewrite, so you can see a rewrite actually got simpler.
- **Grounding rate** — of the sentences a draft wrote, how many are backed by the
  input. A low grounding rate means the model made something up (hallucination).
- **Omission check** — of the important facts in the input, how many the draft
  left out. Omission is tracked *separately* from made-up content: a draft can be
  perfectly grounded and still unsafe because it dropped a medication change or a
  follow-up. One number would hide the other.

Grounding and omission are computed offline with TF-IDF cosine similarity — no
network, no API key. Reading level uses ``textstat``; if ``textstat``'s syllable
data is unavailable in a locked-down offline environment, a small self-contained
estimator computes the same Flesch formulas so the offline path always works.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Similarity at or above this counts a sentence as supported by a source snippet.
DEFAULT_SUPPORT_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# Reading level
# ---------------------------------------------------------------------------

def _count_syllables(word: str) -> int:
    """Heuristic syllable count: vowel groups, minus a silent trailing 'e'."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and not word.endswith(("le", "ie", "ee")) and n > 1:
        n -= 1
    return max(1, n)


def _fallback_reading_level(text: str) -> dict:
    """Flesch-Kincaid grade and reading ease with no external data files."""
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"[A-Za-z]+", text)
    n_sent = max(1, len(sentences))
    n_words = max(1, len(words))
    syllables = sum(_count_syllables(w) for w in words)
    words_per_sent = n_words / n_sent
    syll_per_word = syllables / n_words
    fk = 0.39 * words_per_sent + 11.8 * syll_per_word - 15.59
    ease = 206.835 - 1.015 * words_per_sent - 84.6 * syll_per_word
    poly = sum(1 for w in words if _count_syllables(w) >= 3)
    smog = 1.043 * ((poly * (30 / n_sent)) ** 0.5) + 3.1291
    return {
        "flesch_kincaid_grade": round(fk, 2),
        "flesch_reading_ease": round(ease, 2),
        "smog_index": round(smog, 2),
    }


def reading_level(text: str) -> dict:
    """Reading-level metrics for ``text``.

    Prefers ``textstat``; falls back to the self-contained estimator if
    ``textstat``'s syllable corpus cannot be loaded offline. Both compute the
    same Flesch formulas, so results are comparable.
    """
    text = (text or "").strip()
    if not text:
        return {"flesch_kincaid_grade": 0.0, "flesch_reading_ease": 0.0, "smog_index": 0.0}
    try:
        import textstat

        return {
            "flesch_kincaid_grade": round(float(textstat.flesch_kincaid_grade(text)), 2),
            "flesch_reading_ease": round(float(textstat.flesch_reading_ease(text)), 2),
            "smog_index": round(float(textstat.smog_index(text)), 2),
        }
    except Exception:
        return _fallback_reading_level(text)


def reading_level_change(before: str, after: str) -> dict:
    """Reading level before and after a rewrite, plus the grade delta.

    ``grade_drop`` is positive when the rewrite is easier to read.
    """
    b = reading_level(before)
    a = reading_level(after)
    return {
        "before": b,
        "after": a,
        "grade_drop": round(b["flesch_kincaid_grade"] - a["flesch_kincaid_grade"], 2),
        "got_simpler": a["flesch_kincaid_grade"] < b["flesch_kincaid_grade"],
    }


# ---------------------------------------------------------------------------
# Grounding (is each drafted sentence backed by the input?)
# ---------------------------------------------------------------------------

@dataclass
class SupportResult:
    text: str
    supported: bool
    score: float
    best_source_id: str | None = None


def support_scores(sentences: list[str], sources: list[str],
                   source_ids: list[str] | None = None,
                   threshold: float = DEFAULT_SUPPORT_THRESHOLD) -> list[SupportResult]:
    """For each sentence, the most similar source snippet and whether it clears
    the support threshold. TF-IDF cosine; fully offline."""
    if source_ids is None:
        source_ids = [f"S{i + 1}" for i in range(len(sources))]
    results: list[SupportResult] = []
    if not sources:
        return [SupportResult(text=s, supported=False, score=0.0) for s in sentences]
    vect = TfidfVectorizer().fit(sources + sentences)
    src_mat = vect.transform(sources)
    for sent in sentences:
        sim = cosine_similarity(vect.transform([sent]), src_mat)[0]
        best = int(sim.argmax())
        score = float(sim[best])
        results.append(SupportResult(
            text=sent, supported=score >= threshold, score=round(score, 4),
            best_source_id=source_ids[best] if score > 0 else None))
    return results


@dataclass
class GroundingReport:
    grounding_rate: float
    supported: int
    total: int
    unsupported: list[str] = field(default_factory=list)


def grounding_rate(sentences: list[str], sources: list[str],
                   threshold: float = DEFAULT_SUPPORT_THRESHOLD) -> GroundingReport:
    """Fraction of drafted sentences the input actually supports."""
    scored = support_scores(sentences, sources, threshold=threshold)
    total = len(scored)
    supported = sum(1 for r in scored if r.supported)
    unsupported = [r.text for r in scored if not r.supported]
    rate = supported / total if total else 1.0
    return GroundingReport(grounding_rate=round(rate, 4), supported=supported,
                           total=total, unsupported=unsupported)


# ---------------------------------------------------------------------------
# Omission (which important input facts did the draft leave out?)
# ---------------------------------------------------------------------------

@dataclass
class OmissionReport:
    omission_rate: float
    missed: list[str]
    important_facts: int
    covered: int


def omission_check(important_facts: list[str], output_text: str,
                   threshold: float = DEFAULT_SUPPORT_THRESHOLD) -> OmissionReport:
    """How many important input facts never made it into the output.

    Each fact is one clinically-important item (a diagnosis, a med change, a
    follow-up). A fact counts as covered if some part of the output is similar
    enough to it. Tracked apart from made-up content, on purpose.
    """
    facts = [f for f in important_facts if f.strip()]
    if not facts:
        return OmissionReport(omission_rate=0.0, missed=[], important_facts=0, covered=0)
    out_sentences = [s for s in re.split(r"[.!?]+", output_text) if s.strip()]
    if not out_sentences:
        return OmissionReport(omission_rate=1.0, missed=list(facts),
                              important_facts=len(facts), covered=0)
    vect = TfidfVectorizer().fit(facts + out_sentences)
    out_mat = vect.transform(out_sentences)
    missed = []
    for fact in facts:
        sim = cosine_similarity(vect.transform([fact]), out_mat)[0]
        if float(sim.max()) < threshold:
            missed.append(fact)
    covered = len(facts) - len(missed)
    rate = len(missed) / len(facts)
    return OmissionReport(omission_rate=round(rate, 4), missed=missed,
                          important_facts=len(facts), covered=covered)
