# Model card — guideline-cds

## What it is
A question-answering tool for licensed clinicians that answers only from retrieved
medical-guideline text and attaches a citation to every clinical sentence it keeps.
A sentence it cannot tie to a retrieved source is deleted before the answer is
shown; if nothing survives, the tool abstains with "not enough evidence to answer".

## Intended use
Point-of-care reference and decision support for a licensed clinician who reads the
cited sources and makes the call. It surfaces guideline-concordant options and their
basis. It is not a diagnosis, not a patient-specific directive, and not a medical
device.

## Out of scope
- Autonomous diagnosis or treatment decisions.
- Patient-specific dosing commands.
- Emergency or resuscitation management.
- Any use where the clinician does not independently review the cited sources.

## How it works
1. Retrieve the most relevant guideline snippets (offline: TF-IDF over a small
   built-in example corpus; real use: a stronger retriever over the full corpus).
2. Ask the model to answer using only those snippets, citing each sentence inline.
3. Run a deterministic citation gate: drop any clinical sentence whose citation is
   missing or not in the retrieved context; abstain if nothing survives.
4. Report a statement-attribution rate (fraction of clinical sentences that carry a
   real, in-context citation).

## Data sources
- Offline example corpus: short, paraphrased, illustrative snippets in
  `src/corpus.py`. Not a clinical reference.
- Real use:
  - `epfl-llm/guidelines` (Hugging Face), filtered to its redistributable sources
    (for example CDC, WHO, WikiDoc); per-source licenses apply and must be checked.
  - USPSTF Prevention TaskForce API — US-government public domain; free key.
  - openFDA drug-label API — US-government public domain; open, optional key.
  - CDC / WHO public pages — public domain / respective terms.

## Model
- Offline: a deterministic stand-in that answers strictly from the supplied
  snippets, so tests and demos run with no key and no network.
- Real use: `claude-sonnet-5` via the official `anthropic` SDK when
  `ANTHROPIC_API_KEY` is set.

## Evaluation
`statement_attribution_rate` measures the fraction of an answer's clinical
sentences with a real supporting citation. After the gate, surviving answers score
1.0 by construction; running the metric on the raw model output shows how much the
gate removed. Add held-out clinical Q&A sets and a hallucination benchmark before
any real deployment.

## Limitations and risks
- Retrieval quality bounds answer quality; a thin or mismatched corpus means more
  abstentions.
- The offline model is a stand-in, not a clinical reasoner.
- Guidelines change; live USPSTF/openFDA calls should record their pull date.
- The tool can be wrong or incomplete. The clinician verifies every source.

## Safety framing
Decision support that a clinician independently reviews, not an autonomous system
and not a medical device. The "no source, no answer" rule is enforced in code, not
left to the prompt.
