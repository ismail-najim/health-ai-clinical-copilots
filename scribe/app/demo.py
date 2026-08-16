"""Gradio demo: paste a transcript, get a note with receipts.

For each note line you see the transcript snippet it came from, and any line the
tool could not back up is highlighted so a clinician can catch it before signing.
Runs offline on the built-in stand-in; set ANTHROPIC_API_KEY for real notes.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from src.eval import grounding_rate
from src.scribe import Scribe

EXAMPLE = """Doctor: Hi, what brings you in today?
Patient: I've had a sore throat and a cough for about five days now.
Patient: The cough gets worse at night and I've been a bit feverish.
Doctor: Any trouble breathing or chest pain?
Patient: No chest pain, breathing is fine.
Doctor: Are you taking any medication?
Patient: Just ibuprofen when the throat hurts. I'm allergic to penicillin.
Doctor: Okay. Let's do a rapid strep test and I'll check your throat.
Doctor: If it's strep we'll start an antibiotic that isn't penicillin.
Doctor: Come back in a week if the cough hasn't settled."""


def _render(raw_transcript: str):
    scribe = Scribe()
    note, transcript = scribe.run(raw_transcript)
    report = grounding_rate(note)

    parts = []
    backend = "real model" if os.environ.get("ANTHROPIC_API_KEY") else "offline stand-in"
    parts.append(
        f"<p style='color:#555'>Backend: {backend}. "
        f"Grounded {report.supported}/{report.total} sentences "
        f"({report.grounding_rate:.0%}). "
        f"Flagged {len(note.flagged)} unsupported.</p>"
    )

    for heading, sentences in note.by_section():
        parts.append(f"<h3 style='margin:14px 0 4px'>{heading}</h3>")
        for s in sentences:
            if s.status == "unsupported":
                parts.append(
                    f"<div style='background:#ffe5e5;border-left:4px solid #d33;"
                    f"padding:8px;margin:6px 0'>"
                    f"<b>&#9888; unsupported</b> &mdash; {s.text}"
                    f"<br><small style='color:#a00'>no transcript line backs this up"
                    f" (best match {s.best_similarity:.2f})</small></div>"
                )
            else:
                cites = "; ".join(f"[line {e.line}] &ldquo;{e.text}&rdquo;"
                                  for e in s.evidence)
                parts.append(
                    f"<div style='border-left:4px solid #3a7;padding:8px;margin:6px 0'>"
                    f"{s.text}"
                    f"<br><small style='color:#367'>source: {cites}</small></div>"
                )

    parts.append("<hr><p style='color:#777'><b>Assistive only.</b> A clinician "
                 "reviews the receipts and signs the note. This is not a medical "
                 "device and does not file anything on its own.</p>")
    return "".join(parts)


def build() -> gr.Blocks:
    with gr.Blocks(title="scribe") as demo:
        gr.Markdown("# scribe — visit conversation to note, with receipts\n"
                    "Paste a doctor-patient transcript. Each note line shows the "
                    "transcript snippet it came from; anything unsupported is flagged.")
        with gr.Row():
            inp = gr.Textbox(value=EXAMPLE, lines=16, label="Transcript")
            out = gr.HTML(label="Note")
        btn = gr.Button("Write the note", variant="primary")
        btn.click(_render, inputs=inp, outputs=out)
    return demo


if __name__ == "__main__":
    build().launch()
