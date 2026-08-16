# Model card — documentation

## Overview

An assistive drafting tool for clinical paperwork with three surfaces:

1. **Pre-visit brief** — pulls the key problems, medications, and open items from
   a patient's note snippets into a short, cited brief.
2. **Discharge summary** — drafts a brief hospital course and patient
   instructions from an encounter, and flags any sentence the input does not
   support.
3. **Plain-English rewrite** — rewrites medical text to a patient reading level
   while checking that no safety warning is dropped.

It drafts; a clinician reviews and signs. It is **not** a diagnostic or treatment
device and produces no autonomous output.

## Intended use

- **Users:** clinicians and clinical documentation staff, as a drafting aid.
- **In scope:** producing review-ready drafts that are grounded in supplied
  notes, with grounding and safety checks surfaced for the reviewer.
- **Out of scope:** autonomous documentation, diagnosis, treatment decisions, or
  anything filed or handed to a patient without clinician review and sign-off.

## How it works

Each surface asks a language model for a draft through one interface
(`src/llm.py`). With `ANTHROPIC_API_KEY` set, that is the Anthropic API
(`claude-sonnet-5`); with no key, a deterministic offline stand-in drafts from the
input alone so the tool runs with no network. Grounding, support, and safety
checks (`src/eval.py`, `src/simplify.py`, `src/discharge.py`) run identically
whichever model produced the draft; they use offline TF-IDF similarity and
`textstat` reading-level metrics.

## Data

- **Plain-English rewrite** is built and evaluated on open, credential-free
  corpora: **PLABA** (NLM plain-language adaptation of biomedical abstracts),
  **Med-EASi** (expert-annotated medical simplification), and **Cochrane-auto**
  (aligned Cochrane plain-language summaries; per-source terms). See
  `data/prepare.py`.
- **Pre-visit brief and discharge summary** are normally built on MIMIC-derived
  corpora, which are **credentialed** (PhysioNet) and must run on self-hosted
  models — no protected health information is sent to an external API. The
  shipped offline demo therefore uses the open simplification path plus small
  **synthetic, PHI-free** notes. No credentialed data ships with this tool.

## Metrics

Reported as separate numbers, because they fail in different ways:

- **Grounding rate** — fraction of drafted sentences the input supports. Low
  grounding means made-up content.
- **Omission** — important input facts left out of the draft. Tracked **apart**
  from made-up content; an omitted medication change or follow-up is often the
  bigger risk.
- **Reading level** — Flesch-Kincaid grade (plus reading ease and SMOG) before
  and after a rewrite; target is roughly a 6th-to-8th grade band.
- **Safety preservation** — fraction of safety-critical instructions (danger
  signs, when to seek care, doses/timing, follow-up) that survive a rewrite. A
  dropped warning is a **blocking** failure, reported separately from the reading
  score.

## Limitations

- The offline stand-in is deterministic and simple; it demonstrates the checks,
  not production-quality drafting. Real drafts need a key.
- Grounding, omission, and safety checks use lexical (TF-IDF) similarity, which
  can miss paraphrase or be fooled by shared vocabulary. Treat the flags as a
  prompt to review, not a guarantee.
- Reading-level scores approximate readability; they do not measure comprehension.
- No patient-data handling is included; any real deployment must keep
  MIMIC-derived and other protected data on self-hosted models.

## Safety and human oversight

Assistive only. A clinician reviews and signs every output; the tool is never the
author of record. Grounding and omission are tracked separately, and
over-simplification that drops a safety warning blocks the rewrite. Not a medical
device.
