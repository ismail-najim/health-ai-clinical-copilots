# Architecture

The flow is short and deliberately boring: retrieve, generate under a strict
"cite everything" instruction, then a deterministic gate that deletes anything not
backed by a retrieved source.

```
question
   |
   v
[retriever]  TF-IDF over the corpus -> top-k snippets (or nothing)
   |                                        |
   |  no hits ------------------------------+--> ABSTAIN
   v
[model]  answer using ONLY the snippets, cite every sentence inline
   |
   v
[citation gate]  drop each clinical sentence whose citation is
                 missing or not in the retrieved context
   |                                        |
   |  nothing survives ---------------------+--> ABSTAIN
   v
answer (only cited sentences) + sources + removed-sentence list
   |
   v
[attribution metric]  fraction of clinical sentences that are cited
```

## Files

- `src/llm.py` — the model interface. `BaseLLM.complete(...)` is the one method the
  flow calls. `AnthropicLLM` wraps the official SDK (lazy, so import never needs a
  key). `MockLLM` is a deterministic offline stand-in that answers strictly from the
  snippets in the prompt and tags each sentence with its source id. `default_llm()`
  returns the real client when `ANTHROPIC_API_KEY` is set, otherwise the mock.

- `src/corpus.py` — a small built-in set of example guideline snippets, each with an
  id, title, authority, license, and url. An offline stand-in. In real use you load
  the `epfl-llm/guidelines` corpus here instead.

- `src/retriever.py` — `Retriever`: TF-IDF plus cosine similarity over the corpus
  (scikit-learn, no downloads). Returns up to k snippets above a score floor, so an
  off-topic question returns nothing.

- `src/cds.py` — the main flow. `GuidelineCDS.answer(question)` retrieves, prompts
  the model, runs `citation_gate(...)`, and abstains when there is no support.
  `citation_gate` and the abstain message live here.

- `src/tools.py` — `uspstf_lookup(...)` (USPSTF Prevention TaskForce API) and
  `openfda_label_lookup(...)` (openFDA drug labels), each with an offline-safe
  fallback so tests need no network. Returns records shaped like corpus snippets.

- `src/eval.py` — `statement_attribution_rate(...)`: fraction of an answer's
  clinical sentences that carry a real, in-context citation.

- `app/demo.py` — a Gradio UI: ask a question, get a cited answer or an honest
  refusal, with sources and the list of removed sentences.

- `tests/test_smoke.py` — offline CPU test: retrieval, a cited answer, the abstain
  path, the gate dropping unsupported sentences, and the metric.

## The gate, precisely
A sentence is "clinical" if it is a full assertion (four or more words, not a header
or short framing line). A clinical sentence is kept only if it carries at least one
citation and every citation names an id that was actually retrieved. Everything else
is dropped and recorded. No softening, no rewriting — deleted. If no clinical
sentence survives, the whole answer becomes an abstention.
