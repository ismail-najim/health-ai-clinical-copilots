# Architecture

A patient message runs through a fixed, one-way pipeline:

```
patient message
      |
      v
[ emergency gate ]  --- fires --->  emergency banner, route to on-call, STOP
      |  (no emergency)
      v
[ urgency triage ]  ->  emergency / urgent / routine
      |
      v
[ condense question ]  ->  the single thing the patient is asking
      |
      v
[ draft reply ]  ->  a draft, marked "awaiting clinician approval"
      |
      v
[ clinician approves ]  ->  the only path that marks a reply sent
```

The emergency gate is first and is a plain rule, not a model, so nothing the
model does can let an emergency reach a routine reply. The clinician approval
step is last and is the only way a reply is ever sent.

## Files, one by one

**src/llm.py** — one small interface, `BaseLLM.complete(...)`, with two
implementations. `AnthropicLLM` wraps the official `anthropic` SDK and is lazy,
so importing the module never needs a key or a network. `MockLLM` is a
deterministic stand-in that classifies urgency, condenses a question, and writes
a reply with no key and no network. `default_llm()` returns the real client when
`ANTHROPIC_API_KEY` is set and the mock otherwise, so the same code runs in CI
and in production.

**src/redflags.py** — the deterministic emergency gate. A list of patterns for
chest pain, trouble breathing, stroke signs, suicidal thoughts, severe bleeding,
anaphylaxis, sudden severe headache, and loss of consciousness. `screen(message)`
returns a `RedFlagResult`; when it is an emergency the pipeline stops and returns
"seek emergency care" guidance. Tuned to over-warn: a false alarm is acceptable,
a missed emergency is not.

**src/triage.py** — urgency classification into emergency / urgent / routine.
`BaselineTriage` is a scikit-learn TF-IDF plus logistic-regression model that
trains on a small built-in seed set and runs on CPU in a fraction of a second.
`llm_triage` asks a model through a forced tool call. `triage(...)` picks between
them. The built-in seed set means the baseline works offline with no downloads;
data/prepare.py explains how to train on PMR-Bench instead.

**src/summarize.py** — condense a rambling message to one question. `summarize`
calls the model with a "single question only" instruction, then a cheap
faithfulness check flags a summary that introduces content words the message
never used, since an invented detail would poison the draft.

**src/reply.py** — draft a reply and hold it for approval. `draft_reply` returns
a `Draft` with `sent=False`. The only way it becomes sent is `approve(draft,
clinician)`, which requires a clinician name. No function here sends on its own.

**src/eval.py** — the metrics. `red_flag_recall` is the headline: the fraction
of true emergencies the gate catches, reported on its own with the miss list.
`over_reassurance_rate` is tracked separately and counts drafts that soothe a
patient without pointing them back to care. `triage_accuracy` is reported
alongside, never in place of, red-flag recall.

**data/prepare.py** — fetches MeQSum (open) for the condense-the-question step
and prints access notes for the gated sets: PMR-Bench (urgency, accept terms),
MedRedQA (reply drafting, registration and data-use agreement, research only),
and MedRedFlag (misconception redirect). Nothing here runs during tests.

**app/demo.py** — a Gradio page. Paste a message; if the gate fires you get an
emergency banner and nothing else, otherwise an urgency label, the condensed
question, and a draft reply. A clinician types their name and clicks Approve &
send, which is the only send path in the app.

**tests/test_smoke.py** — offline, CPU, no downloads, all on the stand-in. The
gate catches an emergency and blocks a routine reply; a benign message gets a
label and a draft; the summarizer condenses; a draft is never auto-sent; the
red-flag-recall metric computes; the over-reassurance detector flags the bad
draft.
