"""Fetch the open plain-language simplification datasets.

The plain-English rewrite side of this tool is built and evaluated on open,
credential-free corpora:

- **PLABA** — the NLM Plain Language Adaptation of Biomedical Abstracts:
  professional-to-plain sentence pairs over biomedical abstracts.
  Public release: https://osf.io/rnpmf/ (and the NLM LHNCBC project page).
- **Med-EASi** — expert-annotated medical text simplification with fine-grained
  edits (elaborate / replace / delete). On the Hugging Face Hub as
  ``cbasu/Med-EASi``.
- **Cochrane-auto** — automatically aligned Cochrane plain-language summaries,
  paragraph-level expert-to-plain pairs. Distributed under each source's terms;
  see the Cochrane-auto release repository for access and licensing. Noted here
  as a scale-up source; not downloaded automatically.

The clinical-summary side (pre-visit brief, discharge summary) is normally built
on MIMIC-derived corpora, which are **credentialed** (PhysioNet) and must run on
self-hosted models. This shipped, offline demo therefore uses only the open
simplification path above plus the small synthetic notes written by
``write_synthetic_notes()`` — no credentialed data, no downloads at test time.

Nothing here runs on import. Call the functions explicitly:

    python -m data.prepare --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"

PLABA_SOURCE = "https://osf.io/rnpmf/"  # NLM PLABA public release
MED_EASI_HF = "cbasu/Med-EASi"          # Hugging Face dataset id
COCHRANE_AUTO_NOTE = (
    "Cochrane-auto (aligned Cochrane plain-language summaries) is distributed "
    "under its source's terms — fetch it from the Cochrane-auto release "
    "repository and follow its license before use."
)


def fetch_med_easi(out_dir: Path = RAW_DIR) -> Path:
    """Download Med-EASi from the Hugging Face Hub (needs `datasets` + network)."""
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(MED_EASI_HF)
    target = out_dir / "med_easi.json"
    rows = []
    for split in ds:
        for row in ds[split]:
            rows.append({"split": split, **row})
    target.write_text(json.dumps(rows, indent=2))
    print(f"Med-EASi: wrote {len(rows)} rows -> {target}")
    return target


def fetch_plaba(out_dir: Path = RAW_DIR) -> None:
    """PLABA needs a one-time manual download (license acceptance).

    Grab the release from ``PLABA_SOURCE``, unzip it into ``data/raw/plaba/``,
    and it will sit alongside the other corpora. This function only points you
    there so nothing silently pulls a licensed file.
    """
    print("PLABA is a manual download (accept the NLM terms first):")
    print(f"  {PLABA_SOURCE}")
    print(f"  unzip into: {out_dir / 'plaba'}")


def note_cochrane_auto() -> None:
    print(COCHRANE_AUTO_NOTE)


def write_synthetic_notes(out_dir: Path = DATA_DIR) -> Path:
    """Write a small set of synthetic, PHI-free notes for the offline demo.

    These are invented examples — no real patient data — so the pre-visit brief
    and discharge surfaces have something to run on without any credentialed
    corpus.
    """
    samples = {
        "prebrief_snippets": [
            {"id": "N1", "text": "Type 2 diabetes, diagnosed 2019, HbA1c 8.6% (rising)."},
            {"id": "N2", "text": "Metformin 1000 mg twice daily; dose increased last visit."},
            {"id": "N3", "text": "Hypertension; blood pressure 152/94 today, above target."},
            {"id": "N4", "text": "Reports new numbness in both feet over the last month."},
            {"id": "N5", "text": "Follow up in the diabetes clinic in 4 weeks for repeat labs."},
        ],
        "discharge_encounter": [
            {"id": "E1", "text": "Admitted with community-acquired pneumonia; treated with IV antibiotics."},
            {"id": "E2", "text": "Switched to oral amoxicillin 500 mg three times a day for 7 days."},
            {"id": "E3", "text": "Oxygen levels normal on room air by day 3."},
            {"id": "E4", "text": "Return to the emergency department if you develop chest pain, fever above 39C, or trouble breathing."},
            {"id": "E5", "text": "Follow up with your primary care doctor in one week."},
        ],
        "simplify_text": (
            "The patient was admitted with an acute exacerbation of chronic obstructive "
            "pulmonary disease necessitating administration of bronchodilator therapy, and "
            "the condition improved over the course of the hospitalization. Return to the "
            "emergency department immediately if you develop chest pain or difficulty breathing."
        ),
    }
    out = out_dir / "synthetic_samples.json"
    out.write_text(json.dumps(samples, indent=2))
    print(f"Synthetic PHI-free samples -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch open simplification corpora.")
    ap.add_argument("--med-easi", action="store_true", help="download Med-EASi (HF)")
    ap.add_argument("--plaba", action="store_true", help="print PLABA download instructions")
    ap.add_argument("--synthetic", action="store_true", help="write synthetic demo notes")
    ap.add_argument("--all", action="store_true", help="do everything above")
    args = ap.parse_args()

    if args.all or args.synthetic:
        write_synthetic_notes()
    if args.all or args.plaba:
        fetch_plaba()
    if args.all or args.med_easi:
        fetch_med_easi()
    note_cochrane_auto()
    if not any([args.all, args.med_easi, args.plaba, args.synthetic]):
        ap.print_help()


if __name__ == "__main__":
    main()
