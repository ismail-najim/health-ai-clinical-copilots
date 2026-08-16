# Model card — ehr-agent

## What it is

A small, human-in-the-loop assistant for clinical order entry. Given a patient
id and a clinical intent, it reads the patient's chart (conditions, medications,
labs) and **drafts** an order — a lab/imaging request or a medication request —
with a plain rationale and the record references that justify it. It then hands
the draft to a person, who approves or rejects it. The agent never writes to the
record on its own.

## Intended use

- Reducing the clerical half of order entry and results follow-up: drafting the
  order so a clinician reviews and accepts rather than assembling it by hand.
- A teaching and demonstration tool for the propose-then-confirm pattern, where
  the write is gated behind a human approval as a matter of design.

## Out of scope / not intended use

- Not a medical device and not autonomous clinical decision support. It does not
  diagnose, and it does not act on the patient.
- Not for use on real patient data through an external model API. The demo runs
  on synthetic records; if you connect a real FHIR server, use only synthetic or
  properly de-identified data unless your deployment is cleared for otherwise.
- Its drafts are suggestions, not orders. Nothing it produces reaches a patient
  without a licensed human committing it.

## How it works

- **Model**: `claude-sonnet-5` via the official `anthropic` SDK when
  `ANTHROPIC_API_KEY` is set; a deterministic offline stand-in otherwise, so
  tests and the demo run with no key and no network.
- **Tools**: four read-only tools over the chart (`patient_search`,
  `get_conditions`, `get_medications`, `get_labs`) and one draft-only tool
  (`propose_order`) that returns a proposal with `status: "draft"`.
- **Loop**: bounded (default six steps), so it always terminates.
- **Write path**: separate from the agent and reachable only through a human
  approval. The agent code has no reference to it.

## Safety design

- **No auto-writes.** The boundary is structural: there is no write function on
  the agent's path. `propose_order` builds a draft object; committing it is a
  different, human-triggered action. An eval check asserts zero writes from the
  agent across a run.
- **Justified drafts.** Every proposal must carry a rationale and at least one
  record reference as evidence, checked in `src/eval.py`.
- **Bounded work.** The loop cannot run forever.

## Evaluation

`src/eval.py` runs offline and checks three invariants: no auto-writes, every
proposal is a draft with a rationale and cited evidence, and the loop stays
within its step budget. The smoke tests in `tests/` exercise the same behavior
on the synthetic cohort.

These checks demonstrate the safety invariants; they are not a clinical accuracy
benchmark. Coding correctness, urgency correctness, and task success against a
real order-entry benchmark would be the next measurements before any real use.

## Limitations

- The synthetic store and the offline mock are simple by design; the mock drafts
  a grounded follow-up but does not reason clinically.
- Codes in the synthetic data are illustrative, not validated against a
  terminology service.
- Real-server behavior (search paging, code validation, resource validity) is
  documented but not implemented in the offline default.
