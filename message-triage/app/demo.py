"""Gradio demo: paste a patient message, see triage and a draft you approve.

The flow shown here is the whole point of the tool:

1. The message hits the deterministic emergency gate first. If it fires, you get
   an emergency banner and nothing else — no urgency label, no draft.
2. Otherwise you get an urgency label, the condensed question, and a draft reply.
3. The draft is never sent by the app. A clinician types their name and clicks
   Approve & send; that is the only path that marks a reply as sent.

Runs offline with the built-in stand-in. Set ANTHROPIC_API_KEY for real drafts.
"""
from __future__ import annotations

import gradio as gr

from src.llm import default_llm
from src.redflags import screen
from src.reply import approve, draft_reply
from src.summarize import summarize
from src.triage import default_baseline, llm_triage


def _record_facts(record_text: str):
    return [line.strip() for line in (record_text or "").splitlines() if line.strip()]


def triage_and_draft(message, record_text, use_llm):
    """Run the pipeline for one message. Returns the four display fields."""
    message = (message or "").strip()
    if not message:
        return "Enter a patient message to begin.", "", "", ""

    # 1. Emergency gate, before anything else.
    flag = screen(message)
    if flag.is_emergency:
        banner = (
            f"EMERGENCY GATE FIRED — {flag.category}\n\n{flag.guidance}\n\n"
            "No routine reply is drafted for this message."
        )
        return banner, "emergency", "", "Routed to on-call clinician."

    llm = default_llm()

    # 2. Urgency label (baseline by default, model if requested).
    if use_llm:
        verdict = llm_triage(message, llm=llm)
    else:
        verdict = default_baseline().classify(message)
    urgency_line = f"{verdict.urgency.upper()} — {verdict.reason}"

    # 3. Condense to the real question.
    summary = summarize(message, llm=llm)
    question_line = summary.question
    if not summary.faithful:
        question_line += "  (flagged: may add detail not in the message)"

    # 4. Draft a reply for review. Never sent here.
    drafted = draft_reply(question_line, message, record=_record_facts(record_text), llm=llm)

    return urgency_line, question_line, drafted.body, drafted.status


def approve_and_send(draft_body, clinician):
    """Wired to the Approve button — the only send path in the app."""
    body = (draft_body or "").strip()
    if not body:
        return "There is no draft to send."
    try:
        d = draft_reply("(reviewed by clinician)", body)
        d.body = body
        approve(d, clinician)
    except ValueError as exc:
        return f"Not sent: {exc}"
    return f"SENT and logged. Approved by {d.approved_by} at {d.approved_at}."


with gr.Blocks(title="message-triage") as demo:
    gr.Markdown(
        "# message-triage\n"
        "Sort a patient message by urgency and draft a reply. Emergencies are "
        "caught first by a fixed rule the model can't override, and a clinician "
        "approves every send."
    )
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(label="Patient message", lines=5,
                             placeholder="Paste the patient's message here...")
            rec = gr.Textbox(label="Patient record facts (one per line, optional)",
                             lines=4,
                             placeholder="e.g. On lisinopril 10mg\nLast A1c 6.4")
            use_llm = gr.Checkbox(label="Use the model for urgency "
                                        "(otherwise the scikit-learn baseline)",
                                  value=False)
            go = gr.Button("Triage and draft", variant="primary")
        with gr.Column():
            urg_out = gr.Textbox(label="Urgency / emergency banner", lines=4)
            q_out = gr.Textbox(label="Condensed question", lines=2)
            draft_out = gr.Textbox(label="Draft reply (for clinician review)", lines=8)
            status_out = gr.Textbox(label="Status", lines=1)

    gr.Markdown("### Clinician approval — the only way a reply is sent")
    with gr.Row():
        clinician = gr.Textbox(label="Clinician name", scale=2)
        approve_btn = gr.Button("Approve & send", variant="stop", scale=1)
    send_status = gr.Textbox(label="Send log", lines=1)

    go.click(triage_and_draft, [msg, rec, use_llm],
             [urg_out, q_out, draft_out, status_out])
    approve_btn.click(approve_and_send, [draft_out, clinician], [send_status])


if __name__ == "__main__":
    demo.launch()
