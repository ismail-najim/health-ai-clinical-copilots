# Model card — scribe

## What it is
An assistive clinical scribe. It turns a doctor-patient visit transcript into a
structured note where every sentence is linked back to the transcript lines it
came from, and it flags any sentence the transcript does not support before the
note is shown. A clinician reviews the receipts and signs. It is not a medical
device and it does not file anything on its own.

## Intended use
- Drafting a first-pass visit note from a transcript, for a clinician to check
  and sign.
- Measuring, on open benchmarks, how well a note is grounded in the transcript
  and how much it leaves out.

## Not intended for
- Autonomous documentation. Nothing is filed without a human signature.
- Diagnosis, triage, or treatment decisions.
- Running on protected health information through an external API without
  de-identifying first. Keep real patient data local, or scrub it before it
  leaves the machine.

## How it works
1. **Note generation.** A language model writes a sectioned note from the
   line-numbered transcript. Offline this is a deterministic built-in stand-in;
   with `ANTHROPIC_API_KEY` set it is the real `anthropic` client
   (`claude-sonnet-5`).
2. **Linked evidence.** Each note sentence is matched to its supporting
   transcript lines with offline TF-IDF cosine similarity — no embeddings, no
   network.
3. **Confabulation check.** A sentence whose best supporting line is too weak is
   flagged `unsupported`, so made-up content is surfaced rather than smoothed over.

## Backends
| Backend | When | Needs |
|---|---|---|
| Offline stand-in (`MockLLM`) | no key set | nothing; runs in CI |
| Real client (`AnthropicLLM`) | `ANTHROPIC_API_KEY` set | the `anthropic` SDK and a key |

The optional real speech and clinical model backends named in the approach
(a medical ASR model for audio, a local medical LLM for on-device note
generation) are not bundled here; the tool is text-in and model-agnostic.

## Evaluation
Two numbers, kept separate:
- **Grounding rate** — fraction of note sentences the transcript backs up. Low
  grounding means the scribe invented content.
- **Omission rate** — fraction of reference-note facts the scribe dropped. A
  fully grounded note can still be unsafe if it omits an allergy or a med change.

Benchmarks: ACI-Bench (~207 encounters), MTS-Dialog (~1,700 dialogues),
PriMock57 (57 consults with audio). All open; see `data/prepare.py`.

## Limitations and failure modes
- Grounding uses lexical similarity, not entailment, so a sentence that reuses
  transcript words but distorts their meaning can pass. Treat the receipts as a
  pointer for the reviewer, not a proof.
- Transcription errors upstream can look like confabulation downstream.
- Thresholds are defaults, not calibrated safety guarantees; tune them on your data.

## Safety
Assistive by design. A clinician is the author of record: they review the linked
evidence and the flags, then sign. No autonomous filing. Not a medical device.
