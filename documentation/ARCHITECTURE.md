# Architecture

A plain, file-by-file tour. The shape is simple: one model interface, three
drafting surfaces, and a metrics module they all lean on. The model only ever
produces a *draft*; the grounding and safety checks run the same way regardless
of which model produced it.

## The flow

```
input (note snippets / encounter notes / clinical text)
        |
        v
  a model drafts        <-- src/llm.py  (real Anthropic client, or offline stand-in)
        |
        v
  a check runs          <-- grounding, support, or safety (src/eval.py, per surface)
        |
        v
  draft + flags         --> a clinician reviews and signs
```

## src/llm.py — the model interface and the offline stand-in

One interface, `BaseLLM.complete(...)`, that every surface calls.

- `AnthropicLLM` wraps the official `anthropic` SDK (`claude-sonnet-5`). It is
  lazy — importing the module never needs a key or the network; the client is
  built on first real use.
- `MockLLM` is a deterministic stand-in that drafts each surface from the input
  alone, with no key and no network. It is deliberately faithful: it never
  invents a problem, medication, or warning that isn't in the input, so the
  offline path exercises the honest case and the checks have real text to trace.
- `default_llm()` returns the real client when `ANTHROPIC_API_KEY` is set, and
  the stand-in otherwise.

The stand-in's plain-language helpers (`split_sentences`, `is_safety_sentence`,
a small jargon-to-plain dictionary) also power the simplify surface's safety
check, so warning detection is shared between drafting and verification.

## src/prebrief.py — pre-visit brief

`generate_brief(snippets)` takes note snippets (strings or `{'id','text'}`
dicts), asks the model to group the key points into sections (active problems,
medications, open items), and returns a `Brief`. Each `BriefPoint` carries the
`source_ids` it was drafted from; only ids that match real snippets are kept.
`Brief.citation_rate` and `Brief.uncited` expose whether every point is
attributed.

## src/discharge.py — discharge summary + support check

`generate_discharge(encounter)` drafts two sections — a brief hospital course and
patient instructions — then runs `check_support(...)`, which compares every
drafted sentence against the input with TF-IDF cosine similarity and marks it
`supported` or not. `DischargeDoc.unsupported` and `.has_unsupported` surface any
sentence the input doesn't back — the invented-detail case. `check_support` is
standalone, so it can re-verify a hand-edited draft too.

## src/simplify.py — plain-English rewrite + safety check

`simplify(text)` rewrites clinical text into plainer, shorter sentences, then
returns a `SimplifyResult` with:

- the reading-level change (before/after Flesch-Kincaid, via `src/eval.py`), and
- a `SafetyReport` from `safety_check(...)`, which extracts the safety-critical
  sentences from the source (danger signs, when to seek care, doses, follow-up)
  and confirms each survives in the plain version. A dropped warning sets
  `SafetyReport.blocked` — over-simplification that loses a warning is caught
  independently of how good the reading score is.

## src/eval.py — the metrics, kept separate

- `reading_level(text)` / `reading_level_change(before, after)` — Flesch-Kincaid
  grade, reading ease, SMOG. Uses `textstat`; falls back to a self-contained
  Flesch estimator if `textstat`'s syllable corpus can't load offline.
- `support_scores(...)` / `grounding_rate(...)` — is each drafted sentence backed
  by the input? Low grounding means made-up content.
- `omission_check(...)` — which important input facts did the draft leave out?
  Tracked **separately** from made-up content, because a grounded draft can still
  be unsafe if it dropped a medication change or a follow-up.

All grounding and omission math is offline TF-IDF cosine — no network, no key.

## src/__init__.py — the public surface

Re-exports the functions and dataclasses above so `from src import simplify,
generate_brief, ...` works.

## data/prepare.py — the open corpora

Fetches the credential-free simplification datasets (PLABA, Med-EASi) and notes
Cochrane-auto, and writes small synthetic PHI-free notes for the offline demo.
The clinical-summary corpora are MIMIC-derived and credentialed, so the shipped
demo uses the open simplification path plus synthetic notes. Nothing runs on
import.

## app/demo.py — the Gradio UI

Three tabs — pre-visit brief, discharge summary, plain-English rewrite — each
showing its grounding or safety status. Runs on the offline stand-in by default,
or on the real client when a key is set.

## tests/test_smoke.py — the offline checks

CPU-only, no network, no key. Confirms brief points cite their sources, the
discharge check flags an unsupported sentence, the rewrite gets simpler while
keeping a warning, and the metrics compute.
