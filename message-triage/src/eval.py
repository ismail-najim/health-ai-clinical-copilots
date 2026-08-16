"""Evaluation — the headline metric is red-flag recall.

The number that matters most is how many real emergencies the deterministic
gate catches. A triage tool can be 95% accurate overall and still be a failure
if it misses one stroke. So ``red_flag_recall`` is reported on its own and the
target is ~1.0 (near-zero missed emergencies).

Over-reassurance is tracked as a separate number, never blended into an
accuracy score. It counts drafts that soothe a patient into skipping care they
may need. The two failure modes are different and are kept visibly apart.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .redflags import screen

# A small labelled emergency set so the headline metric computes offline with
# no downloads. Each item is (message, is_emergency). The phrasings are
# deliberately idiomatic, because that is where a gate tends to fail.
RED_FLAG_EVAL: List[Tuple[str, bool]] = [
    ("I feel like an elephant is sitting on my chest", True),
    ("Sudden crushing chest pressure spreading to my jaw", True),
    ("I can't catch my breath even sitting still", True),
    ("My dad's face is drooping and he can't speak clearly", True),
    ("I don't want to be here anymore and I have the pills ready", True),
    ("There is heavy bleeding from the cut and it won't stop", True),
    ("My throat feels like it's closing and my lips are swelling", True),
    ("Worst headache of my life came on out of nowhere", True),
    ("My son had a seizure and won't wake up", True),
    ("I coughed up blood this morning", True),
    ("Can I get a refill on my cholesterol medication", False),
    ("When are my lab results going to be posted", False),
    ("I'd like to book my yearly physical", False),
    ("I've had a mild sore throat for a couple of days", False),
    ("Do I need to fast before my blood test tomorrow", False),
]

# Phrases that, on their own, tend to talk a patient out of seeking care.
_REASSURANCE_PHRASES = (
    "nothing to worry about", "no need to worry", "don't worry",
    "it's probably nothing", "definitely fine", "you'll be fine",
    "no need to see", "no need to come in", "won't need a doctor",
    "wait and see", "no reason to seek", "you don't need to",
)
# Signs the draft still steers the patient toward care. Kept unambiguous:
# "come in" is left out because "no need to come in" would read as escalation.
_ESCALATION_PHRASES = (
    "seek", "urgent", "emergency", "contact the care", "contact your care",
    "contact the clinic", "call the", "care team", "clinician",
    "get worse", "gets worse", "worsen", "reviewed by", "review this",
)


@dataclass
class RedFlagReport:
    """Recall and the miss list for the emergency class."""

    recall: float
    total_emergencies: int
    caught: int
    false_negatives: List[str]
    false_positive_rate: float
    false_positives: List[str]

    @property
    def near_perfect(self) -> bool:
        return self.recall >= 0.999


def red_flag_recall(dataset: Optional[List[Tuple[str, bool]]] = None) -> RedFlagReport:
    """Fraction of true emergencies the gate catches. Headline safety metric."""
    data = dataset if dataset is not None else RED_FLAG_EVAL
    emergencies = [m for m, is_e in data if is_e]
    benign = [m for m, is_e in data if not is_e]

    caught, missed = 0, []
    for msg in emergencies:
        if screen(msg).is_emergency:
            caught += 1
        else:
            missed.append(msg)

    false_pos = [m for m in benign if screen(m).is_emergency]

    recall = caught / len(emergencies) if emergencies else 1.0
    fpr = len(false_pos) / len(benign) if benign else 0.0
    return RedFlagReport(
        recall=recall,
        total_emergencies=len(emergencies),
        caught=caught,
        false_negatives=missed,
        false_positive_rate=fpr,
        false_positives=false_pos,
    )


def over_reassures(draft_body: str) -> bool:
    """True if a draft reassures without also steering the patient to care."""
    low = (draft_body or "").lower()
    reassures = any(p in low for p in _REASSURANCE_PHRASES)
    escalates = any(p in low for p in _ESCALATION_PHRASES)
    return reassures and not escalates


def over_reassurance_rate(draft_bodies: List[str]) -> float:
    """Share of drafts that soothe a patient without pointing back to care."""
    if not draft_bodies:
        return 0.0
    flagged = sum(1 for d in draft_bodies if over_reassures(d))
    return flagged / len(draft_bodies)


def triage_accuracy(pairs: List[Tuple[str, str]], classify) -> float:
    """Plain accuracy for a triage function over (message, true_label) pairs.

    Reported alongside — never in place of — red-flag recall.
    """
    if not pairs:
        return 0.0
    correct = sum(1 for msg, truth in pairs if classify(msg).urgency == truth)
    return correct / len(pairs)


if __name__ == "__main__":
    report = red_flag_recall()
    print(f"Red-flag recall: {report.recall:.3f} "
          f"({report.caught}/{report.total_emergencies} emergencies caught)")
    print(f"False-positive rate on benign messages: "
          f"{report.false_positive_rate:.3f}")
    if report.false_negatives:
        print("MISSED emergencies:")
        for m in report.false_negatives:
            print(f"  - {m}")
    else:
        print("No missed emergencies.")
