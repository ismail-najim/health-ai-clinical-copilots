"""Fetch and note the datasets this tool evaluates against.

Only one dataset here is fully open and downloads with no account:

- MeQSum: 1,000 consumer-health questions paired with a short summary. It is the
  clean, citable set for the "condense the message to its real question" step.

Two others are gated and cannot be pulled automatically:

- PMR-Bench: patient portal messages labelled for urgency. Accept the terms on
  its dataset card before use; it replaces the small seed set in triage.py for a
  real training run.
- MedRedQA: patient question / physician answer pairs, released under a
  registration and data-use agreement for research only. Use it to study reply
  structure during development; do not embed it in a shipped model.

MedRedFlag (a health-misconception "correct, don't affirm" set) is used to probe
the drafter's behaviour on misconceptions; check its terms before downloading.

Nothing here runs during the tests. Run this file directly to fetch MeQSum and
print access notes for the gated sets.
"""
from __future__ import annotations

import os

# Where cached data lands. Kept out of version control.
DATA_DIR = os.path.join(os.path.dirname(__file__), "cache")

MEQSUM_HF_ID = "sumedh/MeQSum"

ACCESS_NOTES = """
Dataset access
--------------
MeQSum (open, CC-BY): downloaded by this script via the `datasets` library.
    Question summarization: condense a message to its real question.

PMR-Bench (gated): patient portal messages with urgency labels. Accept the
    terms on the dataset card, then point triage.py at it to train the urgency
    baseline on real messages instead of the built-in seed set.

MedRedQA (registration + data-use agreement, research only): question/answer
    pairs. Use for reply-structure study during development only. Do not ship a
    model trained on it.

MedRedFlag (check terms): health-misconception redirect set. Use to probe the
    drafter's "correct, don't affirm" behaviour.
"""


def fetch_meqsum(save_dir: str = DATA_DIR):
    """Download MeQSum with the Hugging Face `datasets` library.

    Requires `pip install datasets` and network access. Returns the loaded
    dataset object. Import is local so this module stays importable offline.
    """
    os.makedirs(save_dir, exist_ok=True)
    from datasets import load_dataset  # local import; not needed offline

    ds = load_dataset(MEQSUM_HF_ID)
    ds.save_to_disk(os.path.join(save_dir, "meqsum"))
    print(f"Saved MeQSum to {os.path.join(save_dir, 'meqsum')}")
    return ds


if __name__ == "__main__":
    print(ACCESS_NOTES)
    try:
        fetch_meqsum()
    except Exception as exc:  # network or missing dependency
        print(f"\nCould not fetch MeQSum automatically ({exc}).")
        print("Install the datasets library and retry: pip install datasets")
