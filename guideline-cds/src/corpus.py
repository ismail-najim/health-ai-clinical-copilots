"""A small, built-in set of example guideline snippets.

This is an OFFLINE stand-in so the tool runs, and its tests pass, with no network
and no downloads. Each snippet carries a stable id, a human title, the issuing
authority, a license label, and a URL, so an answer can render provenance for every
line it keeps.

In real use you replace this with the open guideline corpus
`epfl-llm/guidelines` on Hugging Face (filtered to its redistributable sources such
as CDC, WHO, and WikiDoc), plus live lookups against the USPSTF Prevention
TaskForce API and openFDA drug labels (see `src/tools.py`). The retrieval and
citation-gating code does not change when you swap the corpus.

The example text below is short, paraphrased, illustrative guidance for a demo. It
is NOT a clinical reference. Verify every real recommendation against its primary
source before use.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Snippet:
    id: str
    title: str
    authority: str
    license: str
    url: str
    text: str


# Illustrative, paraphrased snippets. Ids are namespaced by kind:
#   G = guideline text, U = USPSTF-style recommendation, F = FDA-label section.
BUILTIN_CORPUS: list[Snippet] = [
    Snippet(
        id="G1",
        title="Hypertension — initial management",
        authority="Example Guideline",
        license="redistributable-example",
        url="https://example.org/guidelines/hypertension",
        text=("For most adults with newly diagnosed uncomplicated hypertension, "
              "lifestyle changes plus a first-line agent such as a thiazide "
              "diuretic, an ACE inhibitor, an ARB, or a calcium channel blocker "
              "are recommended. Beta blockers are not preferred as initial "
              "monotherapy unless there is another indication."),
    ),
    Snippet(
        id="G2",
        title="Type 2 diabetes — first-line pharmacotherapy",
        authority="Example Guideline",
        license="redistributable-example",
        url="https://example.org/guidelines/type-2-diabetes",
        text=("Metformin is the preferred initial glucose-lowering medication for "
              "most adults with type 2 diabetes when it is not contraindicated and "
              "is tolerated. It is combined with lifestyle and dietary changes."),
    ),
    Snippet(
        id="G3",
        title="Community-acquired pneumonia — outpatient antibiotics",
        authority="Example Guideline",
        license="redistributable-example",
        url="https://example.org/guidelines/community-acquired-pneumonia",
        text=("For previously healthy outpatients with community-acquired "
              "pneumonia and no comorbidities, amoxicillin or doxycycline is a "
              "reasonable first-line antibiotic choice. Macrolide monotherapy is "
              "used only where local pneumococcal resistance is low."),
    ),
    Snippet(
        id="U1",
        title="Colorectal cancer screening",
        authority="USPSTF (example grade)",
        license="public-domain-example",
        url="https://example.org/uspstf/colorectal",
        text=("Screening for colorectal cancer is recommended for adults aged 45 "
              "to 75 years (Grade B for ages 45 to 49, Grade A for ages 50 to 75). "
              "Options include stool-based tests and direct visualization such as "
              "colonoscopy."),
    ),
    Snippet(
        id="U2",
        title="Lung cancer screening",
        authority="USPSTF (example grade)",
        license="public-domain-example",
        url="https://example.org/uspstf/lung",
        text=("Annual screening for lung cancer with low-dose CT is recommended "
              "(Grade B) for adults aged 50 to 80 years who have a 20 pack-year "
              "smoking history and currently smoke or have quit within the past 15 "
              "years."),
    ),
    Snippet(
        id="F1",
        title="Metformin — boxed warning (lactic acidosis)",
        authority="FDA drug label (example)",
        license="public-domain-example",
        url="https://example.org/fda/metformin",
        text=("The metformin label carries a boxed warning for lactic acidosis, a "
              "rare but serious metabolic complication. Risk rises with renal "
              "impairment, and the drug should be assessed against kidney function "
              "before and during use."),
    ),
    Snippet(
        id="F2",
        title="Warfarin — bleeding risk",
        authority="FDA drug label (example)",
        license="public-domain-example",
        url="https://example.org/fda/warfarin",
        text=("The warfarin label carries a boxed warning for major or fatal "
              "bleeding. Regular monitoring of the INR is needed, and many drugs, "
              "foods, and conditions change its effect."),
    ),
]


def corpus_as_dicts() -> list[dict]:
    """The corpus as plain dicts, convenient for retrieval and display."""
    return [
        {"id": s.id, "title": s.title, "authority": s.authority,
         "license": s.license, "url": s.url, "text": s.text}
        for s in BUILTIN_CORPUS
    ]
