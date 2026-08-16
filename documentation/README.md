# documentation — pre-visit briefs, discharge summaries, and plain-English rewrites

A small assistant for clinical paperwork. It **drafts** three things — a short
pre-visit brief, a discharge summary, and a plain-English rewrite of medical text
— and a **clinician reviews and signs**. Every line is either grounded in the
input it was given or clearly flagged, so nothing slips through unchecked.

## What you need

- Python 3.11
- No GPU
- Nothing else to run the offline demo: a deterministic stand-in is built in, so
  it works with no API key and no network.
- Set `ANTHROPIC_API_KEY` if you want real model output instead of the stand-in.

## Easiest way to see it work

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

You should see `6 passed`. The tests run fully offline on the built-in stand-in —
no key, no downloads.

## Run it

```bash
python -m app.demo        # Gradio UI: three tabs — pre-brief, discharge, simplify
```

By default this runs on the offline stand-in. For real drafts, set a key first:

```bash
export ANTHROPIC_API_KEY=sk-...
python -m app.demo
```

You can also use the pieces directly:

```python
from src.prebrief import generate_brief
from src.discharge import generate_discharge
from src.simplify import simplify

brief = generate_brief([{"id": "N1", "text": "Type 2 diabetes, uncontrolled."}])
print(brief.render(), brief.citation_rate)
```

## What you'll see

- **Pre-visit brief** — the key problems, medications, and open items pulled from
  a set of note snippets, grouped into sections, with **every point cited** back
  to the snippet it came from.
- **Discharge summary** — a brief hospital course plus patient instructions,
  where any sentence the input does not support is **flagged for review** instead
  of being filed quietly.
- **Plain-English rewrite** — simpler, shorter text with the **reading level
  shown before and after**, and a check that confirms no safety warning was
  dropped.

## If something doesn't work

- `ModuleNotFoundError: src` — run the commands from the project folder so `src`
  is importable (`python -m pytest`, `python -m app.demo`).
- Reading-level numbers still appear even with no internet: `textstat` uses a
  syllable corpus that some locked-down environments can't load; the code falls
  back to a self-contained estimator that computes the same Flesch formulas.
- Tests hang or try to reach the network: they shouldn't — everything runs on
  the offline stand-in. If you set `ANTHROPIC_API_KEY`, unset it to force the
  offline path.

## Safety

This is an **assistive** tool. A clinician reviews and signs; the tool is never
the author of record and is **not a medical device**. The design keeps two honest
habits: every drafted line is grounded in the input or flagged, and made-up
content is tracked separately from **omission** (something important left out — an
omitted medication change or follow-up is often the bigger risk). For the
plain-English rewrite, over-simplification must never drop safety information: a
dropped danger sign, dose, or follow-up **blocks** the rewrite regardless of how
readable it is.

## Learn more

- `ARCHITECTURE.md` — what each file does, top to bottom.
- `model_card.md` — intended use, data, metrics, and limits.
- `config.yaml` — the thresholds and model settings in one place.
- `data/prepare.py` — the open simplification corpora (PLABA, Med-EASi,
  Cochrane-auto) and how to fetch them.
