"""Gradio demo — three tabs: pre-brief, discharge, simplify.

Runs offline on the deterministic stand-in with no key. Set ANTHROPIC_API_KEY
for real model output. Every surface shows its grounding or safety check, because
this is an assistant: it drafts, a clinician reviews and signs.

    python -m app.demo
"""
from __future__ import annotations

import gradio as gr

from src.prebrief import generate_brief
from src.discharge import generate_discharge
from src.simplify import simplify


def run_prebrief(snippets_text: str):
    rows = [r.strip() for r in snippets_text.splitlines() if r.strip()]
    snippets = [{"id": f"N{i + 1}", "text": r} for i, r in enumerate(rows)]
    if not snippets:
        return "Paste one note snippet per line.", ""
    brief = generate_brief(snippets)
    status = f"Every point cited: {brief.citation_rate:.0%}"
    if brief.uncited:
        status += f"  |  uncited points flagged: {len(brief.uncited)}"
    return brief.render(), status


def run_discharge(encounter_text: str):
    rows = [r.strip() for r in encounter_text.splitlines() if r.strip()]
    encounter = [{"id": f"E{i + 1}", "text": r} for i, r in enumerate(rows)]
    if not encounter:
        return "Paste one encounter note per line.", ""
    doc = generate_discharge(encounter)
    n_flagged = len(doc.unsupported)
    if n_flagged:
        status = f"WARNING: {n_flagged} sentence(s) not supported by the input — flagged for review."
    else:
        status = "Every sentence is supported by the input."
    return doc.render(), status


def run_simplify(clinical_text: str):
    if not clinical_text.strip():
        return "Paste some clinical text.", ""
    result = simplify(clinical_text)
    before = result.reading["before"]["flesch_kincaid_grade"]
    after = result.reading["after"]["flesch_kincaid_grade"]
    status = (f"Reading grade {before:.1f} -> {after:.1f}  |  "
              f"warnings preserved {result.safety.preservation_rate:.0%}")
    if result.safety.blocked:
        status += "  |  BLOCKED: a safety warning was dropped — do not release."
    return result.simplified, status


with gr.Blocks(title="documentation — a clinician reviews and signs") as demo:
    gr.Markdown(
        "# documentation\n"
        "Pre-visit briefs, discharge summaries, and plain-English rewrites. "
        "This tool **drafts**; a clinician **reviews and signs**. Every line is "
        "grounded in the input or flagged. Assistive only — not a medical device."
    )

    with gr.Tab("Pre-visit brief"):
        gr.Markdown("Paste note snippets, one per line. Each brief point cites the line it came from.")
        pb_in = gr.Textbox(label="Note snippets (one per line)", lines=8)
        pb_out = gr.Markdown(label="Brief")
        pb_status = gr.Textbox(label="Grounding", interactive=False)
        gr.Button("Draft brief").click(run_prebrief, pb_in, [pb_out, pb_status])

    with gr.Tab("Discharge summary"):
        gr.Markdown("Paste encounter notes, one per line. Any sentence the notes do not support is flagged.")
        dc_in = gr.Textbox(label="Encounter notes (one per line)", lines=8)
        dc_out = gr.Markdown(label="Discharge draft")
        dc_status = gr.Textbox(label="Support check", interactive=False)
        gr.Button("Draft discharge").click(run_discharge, dc_in, [dc_out, dc_status])

    with gr.Tab("Plain-English rewrite"):
        gr.Markdown("Paste clinical text. The rewrite is checked so warnings are never dropped.")
        sp_in = gr.Textbox(label="Clinical text", lines=8)
        sp_out = gr.Textbox(label="Plain-English version", lines=8)
        sp_status = gr.Textbox(label="Reading level + safety", interactive=False)
        gr.Button("Rewrite").click(run_simplify, sp_in, [sp_out, sp_status])


if __name__ == "__main__":
    demo.launch()
