# Architecture

A plain, file-by-file tour. The whole thing is a bounded loop that reads a
patient record and returns a draft order for a human to approve. Nothing writes
to the record on its own.

## The flow

```
patient id + intent
      │
      ▼
 src/agent.run(...)         bounded loop (default 6 steps)
      │
      ├─ model picks a tool ──► read tools ──► src/fhir  (reads only)
      │                          get_conditions / get_medications /
      │                          get_labs / patient_search
      │
      └─ model calls propose_order ──► src/tools.build_draft
                                        returns a DRAFT resource
                                        (status: "draft") — no write
      ▼
 result: proposals + the reads that justify them
      │
      ▼
 app/demo.py               a human sees the draft and clicks
      ├─ Approve ──► store.create_from_proposal   ← the ONLY write, human-triggered
      └─ Reject  ──► discard; the record is unchanged
```

The safety boundary is that the agent path (`src/agent.py`, `src/tools.py`) has
no reference to the store's write function. A draft is just an object; committing
it lives in the store and is only called from the demo's Approve button.

## Files

### `src/llm.py`
The one model interface, `BaseLLM.step(system, messages, tools)`, returning free
text and/or tool calls.
- `AnthropicLLM` wraps the `anthropic` SDK and is lazy — importing the module
  never needs a key or the network. It translates the neutral message history
  into the SDK's block format.
- `MockLLM` is a deterministic offline stand-in. It walks a fixed plan (read
  conditions, then medications, then labs, then draft one order, then stop),
  choosing each step from what has already been called. Its draft is grounded in
  the labs it read, so the offline path always yields a justified proposal.
- `default_llm()` returns the real client when `ANTHROPIC_API_KEY` is set, the
  mock otherwise.

### `src/fhir.py`
`InMemoryStore`: a fake FHIR-shaped patient store held in dictionaries, with a
few synthetic patients (conditions, medications, labs). Reads are open;
`create_from_proposal` is the single write path and counts its writes so a test
can prove the agent never triggered one. The module docstring shows how to swap
the body for HTTP calls (`requests`) against a real FHIR R4 server such as a
self-hosted HAPI FHIR container, keeping the same read/write split.

### `src/tools.py`
The tool schemas (in anthropic format) and their handlers. Four read tools
resolve to store calls; `propose_order` routes to `build_draft`, which assembles
a FHIR-shaped resource with `status: "draft"`. There is no write function in
this module.

### `src/agent.py`
`run(patient_id, intent, ...)`: the bounded loop. It feeds the model the intent,
runs read tools immediately, collects any `propose_order` calls as drafts, and
stops when the model stops calling tools or the step budget is spent. It returns
the proposals, the reads that justify them, the step count, and a write count
(always zero from this path).

### `src/eval.py`
Three deterministic checks — no auto-writes, every proposal is a draft with a
rationale and cited evidence, and the loop is bounded — plus `run_checks()` that
runs the agent offline and reports. Runnable as `python -m src.eval`.

### `app/demo.py`
A Gradio UI: pick a patient, enter an intent, see what the agent read and the
draft it proposes, then Approve (commit) or Reject (discard). Approve is the only
write.

### `tests/test_smoke.py`
Offline, CPU-only, no network. Confirms the agent reads then drafts, the write
gate holds (nothing auto-written), drafts carry rationale and evidence, the loop
is bounded, the eval checks pass, and approval is the only write path.

### `config.yaml`
Documents the knobs: model name, step budget, FHIR backend and base URL, and the
safety flags the eval enforces.
