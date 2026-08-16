"""A small Gradio demo: pick a patient, see the agent's reading and its draft,
then Approve or Reject. Approve is the only thing that writes; Reject discards.

Run it:  python -m app.demo
It works offline with the built-in mock and synthetic patients. Set
ANTHROPIC_API_KEY to use the real model instead.
"""
from __future__ import annotations

import json

import gradio as gr

from src.agent import run
from src.fhir import default_store
from src.llm import default_llm

# One store for the session, so an approved order actually lands somewhere the
# demo can show. Each agent run reads it; only the Approve button writes to it.
STORE = default_store()
LLM = default_llm()

PATIENT_CHOICES = [f"{p['id']} — {p['name']}" for p in STORE.search_patients()]


def _pid(choice: str) -> str:
    return choice.split(" — ", 1)[0].strip()


def _format_reads(reads: list) -> str:
    lines = ["## What the agent read\n"]
    for r in reads:
        lines.append(f"**{r['tool']}**")
        items = r["result"]
        if isinstance(items, list):
            for it in items:
                ref = it.get("reference", "")
                disp = it.get("display") or it.get("name", "")
                extra = ""
                if "value" in it:
                    extra = f" = {it['value']} {it.get('unit', '')} " \
                            f"(flag {it.get('interpretation', 'N')})"
                lines.append(f"- {disp}{extra}  \n  `{ref}`")
        lines.append("")
    return "\n".join(lines)


def _format_proposal(p: dict) -> str:
    coding = (p.get("code") or p.get("medicationCodeableConcept") or {}).get(
        "coding", [{}])[0]
    rationale = (p.get("reasonCode") or [{}])[0].get("text", "")
    evidence = [s.get("reference", "") for s in p.get("supportingInfo", [])]
    return (
        f"## Proposed order (DRAFT — not written)\n\n"
        f"- **Type**: {p.get('resourceType')}\n"
        f"- **Status**: `{p.get('status')}`\n"
        f"- **Priority**: {p.get('priority')}\n"
        f"- **What**: {coding.get('display', '')} "
        f"(`{coding.get('system', '')}` {coding.get('code', '')})\n"
        f"- **For**: {p.get('subject', {}).get('reference', '')}\n\n"
        f"**Rationale**: {rationale}\n\n"
        f"**Evidence read**: {', '.join(e for e in evidence if e)}"
    )


def draft_order(patient_choice: str, intent: str):
    """Run the agent and show its reading plus the draft, then reveal the gate."""
    result = run(_pid(patient_choice), intent, llm=LLM, store=STORE)
    if not result["proposals"]:
        return (result.get("final_text", "No order proposed."), "", None,
                gr.update(visible=False), "")
    proposal = result["proposals"][0]
    reading = _format_reads(result["reads"])
    draft = _format_proposal(proposal)
    note = (f"_The agent took {result['steps']} step(s) and wrote nothing to the "
            f"record (writes so far this session: {STORE.write_count})._")
    return reading, draft, proposal, gr.update(visible=True), note


def approve(proposal: dict):
    if not proposal:
        return "Nothing to approve."
    committed = STORE.create_from_proposal(proposal)
    return (f"Approved and committed as **{committed['id']}** "
            f"(status `{committed['status']}`). This was the only write, and a "
            f"human triggered it.")


def reject(proposal: dict):
    return "Rejected. The draft was discarded; the record is unchanged."


def build_app():
    with gr.Blocks(title="ehr-agent") as app:
        gr.Markdown(
            "# ehr-agent\n"
            "Reads a patient record and **drafts** an order. It never writes on "
            "its own — you approve or reject every proposal."
        )
        with gr.Row():
            patient = gr.Dropdown(PATIENT_CHOICES, label="Patient",
                                  value=PATIENT_CHOICES[0])
            intent = gr.Textbox(label="Clinical intent",
                                value="Follow up on the most recent abnormal lab.")
        go = gr.Button("Read chart and draft order", variant="primary")

        reading = gr.Markdown()
        draft = gr.Markdown()
        note = gr.Markdown()
        proposal_state = gr.State()

        with gr.Row(visible=False) as gate:
            approve_btn = gr.Button("Approve (commit)", variant="primary")
            reject_btn = gr.Button("Reject (discard)")
        outcome = gr.Markdown()

        go.click(draft_order, [patient, intent],
                 [reading, draft, proposal_state, gate, note])
        approve_btn.click(approve, [proposal_state], [outcome])
        reject_btn.click(reject, [proposal_state], [outcome])
    return app


if __name__ == "__main__":
    build_app().launch()
