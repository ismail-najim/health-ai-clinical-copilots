"""The bounded read-then-draft loop.

``run(...)`` gives the model a patient id and a clinical intent, lets it read the
chart through the read tools, and collects any drafts it proposes. The loop is
capped at ``max_steps`` so it always terminates. The result is a set of
proposals plus the exact record references the agent read — never a change to the
record. Turning a proposal into a real order is a separate, human step.

The safety boundary is structural, not a matter of prompting: this module has no
reference to the store's write path. A ``propose_order`` call is routed to
``build_draft`` and comes back as a draft object. There is simply no code here
that could commit one.
"""
from __future__ import annotations

import json

from .fhir import InMemoryStore, default_store
from .llm import BaseLLM, default_llm
from .tools import TOOL_SCHEMAS, DRAFT_TOOLS, build_draft, run_read_tool

SYSTEM = (
    "You are an assistant that helps with clinical order entry. Read the "
    "patient's chart using the read tools (patient_search, get_conditions, "
    "get_medications, get_labs), then draft the requested order(s) with "
    "propose_order. You never write to the record; you only propose, and a "
    "clinician approves. For every proposed order, give a plain rationale and "
    "cite the exact record references (conditions, labs, medications) that "
    "justify it. Use LOINC codes for lab/imaging orders and RxNorm for "
    "medications. If the chart lacks what you need, say so instead of guessing."
)


def run(patient_id: str, intent: str, *, llm: BaseLLM = None,
        store: InMemoryStore = None, max_steps: int = 6) -> dict:
    """Read the chart and draft order(s) for the given intent.

    Returns a dict with the proposals (drafts awaiting approval), the reads that
    justify them, the number of steps taken, and the agent's closing note.
    Nothing in this function writes to ``store``.
    """
    llm = llm or default_llm()
    store = store or default_store()

    messages = [{"role": "user",
                 "content": f"Patient: {patient_id}\nIntent: {intent}"}]
    proposals: list = []
    reads: list = []
    steps = 0
    final_text = ""

    for steps in range(1, max_steps + 1):
        result = llm.step(system=SYSTEM, messages=messages, tools=TOOL_SCHEMAS)
        messages.append({"role": "assistant", "text": result.text,
                         "tool_calls": result.tool_calls})

        if not result.tool_calls:
            final_text = result.text
            break

        for tc in result.tool_calls:
            if tc.name in DRAFT_TOOLS:
                # A proposal, not a write. build_draft returns a draft object;
                # there is no store write anywhere on this path.
                draft = build_draft(tc.input)
                proposals.append(draft)
                content = json.dumps({"status": "drafted_for_review",
                                      "resourceType": draft["resourceType"]})
            else:
                output = run_read_tool(store, tc.name, dict(tc.input))
                reads.append({"tool": tc.name, "args": dict(tc.input),
                              "result": output})
                content = json.dumps(output)[:6000]
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "name": tc.name, "content": content})

    return {
        "patient_id": patient_id,
        "intent": intent,
        "proposals": proposals,
        "reads": reads,
        "steps": steps,
        "max_steps": max_steps,
        "final_text": final_text,
        "write_count": store.write_count,  # 0 — the agent never writes
    }
