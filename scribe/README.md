# scribe — turn a visit conversation into a note (with receipts)

scribe turns a doctor-patient transcript into a clinical note where every line is
linked back to what was actually said, and it flags anything it can't back up
before you see the note. It is assistive: a clinician reviews the receipts and
signs — nothing is filed on its own.

## What you need

- Python 3.11
- No GPU
- Runs fully offline with a built-in stand-in (no API key), or set
  `ANTHROPIC_API_KEY` to write real notes with the `anthropic` client.

## Easiest way to see it work

```bash
pip install anthropic scikit-learn numpy gradio pytest
python -m pytest tests/ -q
```

You should see `N passed`. That run is offline: no key, no downloads, CPU only.
It sends a transcript through the whole pipeline, confirms each note line maps to
transcript lines, checks that an obviously made-up line gets flagged, and checks
that the grounding and omission numbers compute.

## Run it

**Get the datasets (optional, for evaluation):**

```bash
python data/prepare.py          # fetches ACI-Bench + MTS-Dialog text
```

PriMock57 is the one with audio; `prepare.py` prints the `git clone` command for
it. All three datasets are open.

**Try the demo:**

```bash
python app/demo.py              # opens a local Gradio page
```

Paste a transcript, get the note back with each line's source snippet and any
unsupported line highlighted.

**Use a real model instead of the offline stand-in:**

```bash
export ANTHROPIC_API_KEY=sk-...
python app/demo.py
```

With the key set, notes are written by the real client; without it, everything
still runs on the built-in stand-in.

## What you'll see

- **The note**, organized into sections (chief complaint, history, medications,
  plan).
- **Each line's source** — the transcript snippet, with its line number, that the
  line came from.
- **Flagged lines** — any note sentence the transcript does not support is
  highlighted, so you can catch a made-up statement before signing.

Two numbers, kept separate on purpose:

- **Grounding rate** — how many note lines the transcript backs up. Low grounding
  means the scribe made something up.
- **Omission rate** — how many facts from a reference note the scribe left out. A
  perfectly grounded note can still be unsafe if it drops an allergy or a med
  change, so this is tracked as its own number, not folded into grounding.

## If something doesn't work

- **`ModuleNotFoundError`** — install the requirements:
  `pip install anthropic scikit-learn numpy gradio pytest`.
- **Tests want a key or network** — they should not. They run on the offline
  stand-in; make sure `ANTHROPIC_API_KEY` is unset if you want to confirm the
  offline path.
- **The demo won't open** — Gradio prints a local URL in the terminal; open that.
- **Real notes look empty** — check the key is exported in the same shell and
  that the account has access to the model in `config.yaml`.

## Safety

Assistive only. scribe drafts a note and shows its receipts; a clinician reviews
the linked evidence and the flags, then signs. It does not file anything on its
own and it is not a medical device. Keep real patient data local, or de-identify
it before sending anything to an external model.

## Learn more

- `ARCHITECTURE.md` — how the pipeline fits together, file by file.
- `model_card.md` — intended use, backends, evaluation, limitations.
- `config.yaml` — thresholds and dataset pointers you can tune.
