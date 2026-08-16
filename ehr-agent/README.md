# ehr-agent — reads a patient record and drafts orders (a human always approves)

It reads a patient's chart and drafts an order — a lab, a medication, a
follow-up — with the reasoning and the evidence behind it. It never writes to the
record on its own: it proposes, and a person approves or rejects every action.

## What you need

- Python 3.11. No GPU.
- Runs fully offline with a built-in stand-in model and fake patient data — no
  API key, no network.
- Optional: set `ANTHROPIC_API_KEY` to use the real model for the drafting.

```bash
pip install -r requirements.txt
```

## Easiest way to see it work

Run the tests. They run offline with the built-in stand-in and no key:

```bash
python -m pytest tests/ -q
```

You should see something like `6 passed`. That exercises the whole promise: the
agent reads the fake chart, drafts an order, writes nothing on its own, stays
inside its step budget, and the safety checks pass.

You can also run the checks directly:

```bash
python -m src.eval
```

## Run it

Launch the demo:

```bash
python -m app.demo
```

Offline, it uses the built-in stand-in and the synthetic patients. To use the
real model instead, set a key first:

```bash
export ANTHROPIC_API_KEY=sk-...
python -m app.demo
```

To point at a real FHIR server (for example a self-hosted HAPI FHIR container,
or a public R4 test server), keep the read/write split and swap the store body
for HTTP calls — `src/fhir.py` shows exactly how, and `config.yaml` has the base
URL. Load synthetic patients with Synthea, which exports FHIR R4 bundles.

## What you'll see

Pick a patient and enter an intent (like "follow up on the abnormal lipid
panel"). The agent shows what it read from the chart, then a **proposed order**
marked `draft` — with its type, priority, code, rationale, and the exact record
references it used as evidence. Below it are two buttons: **Approve** and
**Reject**.

This is propose-then-confirm, plainly: the agent only ever produces a proposal.
The order is not real until you click Approve, which is the one and only place
anything gets written. Reject discards the draft and the record stays unchanged.

## If something doesn't work

- `python -m pytest` should pass with no key and no network. If imports fail,
  make sure you are running from the repo root so `src` and `app` are importable.
- The demo needs `gradio` (in `requirements.txt`); the tests and the eval do not.
- If a real model call errors, check that `ANTHROPIC_API_KEY` is set and valid.
  With no key, everything falls back to the offline stand-in automatically.

## Safety

This is an assistive tool, not an autonomous one. It drafts; a human approves
every action. The no-write rule is built into the design, not just asked of the
model: the agent's code has no path to write the record, so a draft can only
become an order when a person clicks Approve. It is not a medical device and does
not make clinical decisions. Use synthetic or properly de-identified data.

## Learn more

- `ARCHITECTURE.md` — a plain, file-by-file tour.
- `model_card.md` — intended use, limitations, and the safety design.
- `config.yaml` — the knobs (model, step budget, FHIR backend).
- Background: Synthea (open synthetic patient records), HAPI FHIR (open FHIR R4
  servers), and the CDS Hooks propose-then-confirm pattern.
