"""Gradio demo: ask a clinical question, get a cited answer or an honest refusal.

Offline (no key) it uses the built-in example corpus and the deterministic
stand-in model. Set ANTHROPIC_API_KEY to use the real model; wire the full
`epfl-llm/guidelines` corpus and the USPSTF / openFDA lookups for real use.

    python -m app.demo
"""
from __future__ import annotations

import gradio as gr

from src.cds import GuidelineCDS
from src.eval import statement_attribution_rate

copilot = GuidelineCDS()


def _render(question: str) -> str:
    if not question or not question.strip():
        return "Enter a clinical question."
    result = copilot.answer(question)

    if result.abstained:
        parts = [f"**{result.answer}**",
                 "",
                 "No retrieved guideline text supported an answer, so the tool "
                 "declined rather than guess."]
        if result.dropped:
            parts.append("\n**Removed by the citation gate:**")
            parts += [f"- ({d['reason']}) {d['sentence']}" for d in result.dropped]
        return "\n".join(parts)

    valid_ids = {h["id"] for h in result.retrieved}
    metric = statement_attribution_rate(result.answer, valid_ids)
    header = ""
    if metric["rate"] is not None:
        header = f"**Statement attribution: {metric['rate']:.0%} "
        header += f"({metric['supported']}/{metric['total']} sentences cited)**\n\n"

    body = [header + result.answer, "", "**Sources**"]
    for s in result.sources:
        grade = f" (Grade {s['grade']})" if s.get("grade") else ""
        body.append(f"- [{s['id']}] {s['title']} — {s['authority']}{grade} "
                    f"— {s['license']} — {s['url']}")

    if result.dropped:
        body.append("\n**Removed by the citation gate (no source, no answer)**")
        body += [f"- ({d['reason']}) {d['sentence']}" for d in result.dropped]

    body.append("\n_Decision support for a licensed clinician who verifies the "
                "sources. Not a diagnosis and not a medical device._")
    return "\n".join(body)


demo = gr.Interface(
    fn=_render,
    inputs=gr.Textbox(label="Clinical question",
                      placeholder="e.g. first-line management of uncomplicated hypertension?"),
    outputs=gr.Markdown(),
    title="guideline-cds — answers from medical guidelines, with citations",
    description=("Answers only from retrieved guideline text and cites every line. "
                 "If it cannot find support, it refuses instead of guessing. "
                 "Decision support for clinicians, not a diagnosis."),
)


if __name__ == "__main__":
    demo.launch()
