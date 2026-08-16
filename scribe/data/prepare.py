"""Fetch the open scribe datasets this tool evaluates against.

All three are public and free to obtain. This script pulls the two text
datasets from their GitHub homes and prints how to get the audio one, which is
larger and lives behind a git clone.

Datasets
--------
ACI-Bench
    ~207 doctor-patient encounters with reference notes. The largest open
    scribe benchmark; good for note-generation and grounding/omission metrics.
    Home: https://github.com/wyim/aci-bench  (open license, see the repo)

MTS-Dialog
    ~1,700 short doctor-patient dialogues paired with summaries. Handy for
    quick end-to-end checks and for tuning.
    Home: https://github.com/abachaa/MTS-Dialog  (open license, see the repo)

PriMock57
    57 mock consultations WITH AUDIO, plus transcripts and reference notes.
    The only one with audio, so it is what you use to test a real speech
    front-end end to end. It is a git clone rather than a single file.
    Home: https://github.com/babylonhealth/primock57  (CC-BY, see the repo)

Run
---
    python data/prepare.py                # fetch ACI-Bench + MTS-Dialog text
    python data/prepare.py --dest ./data  # choose the download folder
"""
from __future__ import annotations

import argparse
import os
import urllib.request

# Raw files on each project's default branch. If a repo reorganizes, clone the
# GitHub home listed in the module docstring instead.
TEXT_SOURCES = {
    "aci-bench": [
        ("https://raw.githubusercontent.com/wyim/aci-bench/main/data/challenge_data/"
         "train.csv", "aci_bench_train.csv"),
        ("https://raw.githubusercontent.com/wyim/aci-bench/main/data/challenge_data/"
         "valid.csv", "aci_bench_valid.csv"),
    ],
    "mts-dialog": [
        ("https://raw.githubusercontent.com/abachaa/MTS-Dialog/main/Main-Dataset/"
         "MTS-Dialog-TrainingSet.csv", "mts_dialog_train.csv"),
        ("https://raw.githubusercontent.com/abachaa/MTS-Dialog/main/Main-Dataset/"
         "MTS-Dialog-ValidationSet.csv", "mts_dialog_valid.csv"),
    ],
}

PRIMOCK57_CLONE = "git clone https://github.com/babylonhealth/primock57"


def fetch(dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    for name, files in TEXT_SOURCES.items():
        print(f"\n{name}:")
        for url, fname in files:
            out = os.path.join(dest, fname)
            if os.path.exists(out):
                print(f"  have {fname}")
                continue
            try:
                print(f"  fetching {fname} ...")
                urllib.request.urlretrieve(url, out)
                print(f"  saved {out}")
            except Exception as exc:  # network is optional; the tool runs offline
                print(f"  could not fetch {fname}: {exc}")
                print(f"  clone the project home instead (see the docstring).")

    print("\nPriMock57 (audio + transcripts + notes):")
    print("  This one has audio, so fetch it with git:")
    print(f"    {PRIMOCK57_CLONE}")
    print("  The audio lets you test a real speech front-end feeding the scribe.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch open scribe datasets.")
    ap.add_argument("--dest", default=os.path.join(os.path.dirname(__file__), "downloads"),
                    help="download folder (default: data/downloads)")
    args = ap.parse_args()
    fetch(args.dest)


if __name__ == "__main__":
    main()
