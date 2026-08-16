# Architecture

Text in, grounded-and-checked note out. Three stages, all runnable offline.

```
raw transcript
   |
parse_transcript ......... line-indexed, speaker-tagged utterances
   |
[1] generate_note ........ model writes a sectioned note (forced tool)
   |
[2] link_and_check ....... TF-IDF cosine links each sentence to its lines
   |                       and flags any sentence with weak support
   v
Note  ->  eval: grounding_rate (made-up?)  +  omission_rate (dropped?)
   |
(demo) clinician reviews receipts + flags, then signs
```

## File by file

### `src/llm.py`
The one model interface, `BaseLLM.complete(...)`, plus `LLMResult`.
- `AnthropicLLM` — real client over the `anthropic` SDK, built lazily so importing
  never needs a key.
- `MockLLM` — deterministic offline stand-in that writes a plausible,
  transcript-grounded note with no network. Powers tests and the offline demo.
- `default_llm()` — real client if `ANTHROPIC_API_KEY` is set, else the mock.

### `src/scribe.py`
The pipeline. `parse_transcript` builds line-indexed utterances. `Scribe` runs
three stages: `generate_note` (stage 1, forced `write_note` tool), and
`link_and_check` (stages 2 and 3 — TF-IDF cosine linking plus the confabulation
flag). Data types: `TranscriptLine`, `Evidence`, `NoteSentence`, `Note`.
`Scribe.run` does the whole thing from raw text.

### `src/eval.py`
The two separate metrics. `grounding_rate` reads the confabulation flags off a
note (how much was made up). `omission_rate` compares the note against a
reference note (how much was dropped), matching facts with TF-IDF cosine.

### `data/prepare.py`
Fetches the open text datasets (ACI-Bench, MTS-Dialog) from their public GitHub
homes and prints the git-clone command for PriMock57, the one with audio.

### `app/demo.py`
Gradio UI: paste a transcript, see the note with each line's source snippet and
any unsupported line highlighted, plus the assistive/human-signs framing.

### `tests/test_smoke.py`
Offline end-to-end, CPU, no key, no downloads. Checks the note comes back, the
evidence maps to real transcript lines, a made-up sentence gets flagged, and both
metrics compute.

## Design choices
- **Offline by default.** No key, no downloads, no GPU to run or test. The real
  client is a drop-in when a key is present.
- **Lexical, not neural, for grounding.** TF-IDF cosine keeps the offline path
  free of heavy model downloads while still tracing each sentence to its source.
- **Two error axes, never merged.** Grounding (made-up content) and omission
  (dropped content) are different failures and are reported as different numbers.
- **Human on the write.** The tool drafts and flags; a clinician signs. Nothing
  is filed autonomously.
