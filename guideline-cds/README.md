# guideline-cds — answers from medical guidelines, with citations (or it says "I don't know")

A question-answering tool for clinicians that answers **only** from real medical
guideline text and puts a citation on **every** line it keeps. If it cannot find a
source to back a claim, it deletes that line — and if nothing is left, it refuses
rather than guess. It is a helper for a licensed clinician who reads the sources and
decides, not a diagnosis.

## What you need
- Python 3.11, no GPU.
- Runs fully offline out of the box: a small built-in example corpus plus a
  deterministic stand-in model, so nothing downloads and no key is needed.
- For real use: set `ANTHROPIC_API_KEY` to use the real model, and load the full
  open guideline corpus (`epfl-llm/guidelines` on Hugging Face, filtered to its
  redistributable sources) plus the live USPSTF and openFDA lookups.

## Easiest way to see it work
```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```
You should see `7 passed`. This runs offline, with no key, on CPU.

## Run it
The Gradio demo (offline by default):
```bash
python -m app.demo
```
Ask a question like "first-line management of uncomplicated hypertension" and you
get a cited answer; ask something the corpus does not cover and you get an honest
refusal.

To switch to real use:
- Set your key: `export ANTHROPIC_API_KEY=...` (the tool then uses `claude-sonnet-5`
  via the `anthropic` SDK; without a key it uses the offline stand-in).
- Load the full corpus: replace the built-in snippets in `src/corpus.py` with the
  `epfl-llm/guidelines` dataset, keeping only its redistributable sources (for
  example CDC, WHO, WikiDoc).
- Turn on the live lookups in `src/tools.py`: `uspstf_lookup(...)` for
  screening/prevention recommendations with their letter grade (needs a free
  `USPSTF_KEY`) and `openfda_label_lookup(...)` for FDA drug-label sections
  (openFDA is open; `OPENFDA_KEY` is optional and only raises the rate limit). Both
  fall back to offline examples so tests never need the network.

## What you'll see
A short answer where every clinical sentence ends with a citation id like `[G1]`,
followed by the list of sources those ids point to (title, authority, license, url,
and a USPSTF grade when present). When a sentence has no valid source, it does not
get softened — it is removed, and the removed lines are shown in their own list so
you can see what was cut. The rule is plain: **no source, no answer.** If the tool
retrieves nothing relevant, or if nothing survives the citation check, it replies
"not enough evidence to answer" instead of making something up.

## If something doesn't work
- `ModuleNotFoundError`: run from the repo root, and `pip install -r requirements.txt`.
- Tests want network or a key: they should not — the suite forces the offline
  stand-in model and the built-in corpus. If you changed `src/corpus.py`, make sure
  it still returns snippets.
- Every question abstains: your query may not match the small example corpus. Try a
  topic it covers (hypertension, type 2 diabetes, pneumonia, colorectal or lung
  screening, metformin, warfarin), or lower `retrieval.min_score` in `config.yaml`.
- The demo will not launch: check that `gradio` installed cleanly; the core tool and
  tests do not need it.

## Safety
This is decision support for a licensed clinician who verifies the cited sources and
makes the decision. It is not autonomous, it does not diagnose a specific patient,
it does not issue dosing commands, and it is not a medical device. The "no source,
no answer" rule is enforced in code — a claim without a retrieved citation is
deleted, not trusted. Always confirm recommendations against their primary source;
the built-in example corpus is illustrative only.

## Learn more
- `ARCHITECTURE.md` — the flow and a file-by-file map.
- `model_card.md` — intended use, data sources and licenses, limitations.
- `config.yaml` — model, retrieval, and source settings.
