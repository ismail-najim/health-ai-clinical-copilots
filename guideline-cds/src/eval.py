"""Attribution metric: how much of an answer is actually backed by a source.

`statement_attribution_rate` is the fraction of an answer's clinical sentences that
carry a citation to an id that was really retrieved. After the citation gate has
run, a well-behaved answer scores 1.0 by construction: every surviving clinical
sentence is cited and in-context. Running the metric on the raw (pre-gate) model
output shows how much the gate had to remove.
"""
from __future__ import annotations

import re

_CITE = re.compile(r"\[([A-Za-z0-9]+)\]")
_FRAMING = re.compile(
    r"^(options|consider|note|summary|in summary|the clinician|this is not|"
    r"decision support)\b", re.I)


def _is_clinical(sentence: str) -> bool:
    s = sentence.strip()
    if not s or s.endswith(":"):
        return False
    if _FRAMING.match(s):
        return False
    return len(s.split()) >= 4


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def statement_attribution_rate(answer_text: str, valid_ids: set[str]) -> dict:
    """Return {'rate', 'supported', 'total'} over the clinical sentences.

    A clinical sentence is 'supported' when it carries at least one citation and
    every citation it carries names an id in `valid_ids`. Answers with no clinical
    sentences (a pure abstain) return rate None.
    """
    clinical = [s for s in _split_sentences(answer_text) if _is_clinical(s)]
    total = len(clinical)
    if total == 0:
        return {"rate": None, "supported": 0, "total": 0}
    supported = 0
    for s in clinical:
        cited = set(_CITE.findall(s))
        if cited and cited.issubset(valid_ids):
            supported += 1
    return {"rate": supported / total, "supported": supported, "total": total}
