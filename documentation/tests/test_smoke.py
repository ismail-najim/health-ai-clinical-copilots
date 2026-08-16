"""Offline smoke tests — CPU only, no network, no API key.

Every test runs on the deterministic MockLLM, so `pytest` passes out of the box.
They check the promises that matter: brief points cite their sources, the
discharge check flags an unsupported sentence, the rewrite gets simpler while
keeping a warning, and the metrics compute.
"""
from __future__ import annotations

from src.llm import MockLLM
from src.prebrief import generate_brief
from src.discharge import (
    DischargeDoc,
    DischargeSection,
    DischargeSentence,
    check_support,
    generate_discharge,
)
from src.simplify import simplify, safety_check
from src.eval import grounding_rate, omission_check, reading_level


MOCK = MockLLM()

NOTE_SNIPPETS = [
    {"id": "N1", "text": "Type 2 diabetes, diagnosed 2019, currently uncontrolled."},
    {"id": "N2", "text": "Metformin 1000 mg twice daily; dose increased last visit."},
    {"id": "N3", "text": "Hypertension, blood pressure elevated at 152/94 today."},
    {"id": "N4", "text": "Follow up in the diabetes clinic in 4 weeks for repeat labs."},
]

ENCOUNTER = [
    {"id": "E1", "text": "Admitted with community-acquired pneumonia, treated with antibiotics."},
    {"id": "E2", "text": "Started on amoxicillin 500 mg three times a day for 7 days."},
    {"id": "E3", "text": "Return to the emergency department if you develop chest pain or trouble breathing."},
    {"id": "E4", "text": "Follow up with your primary care doctor in one week."},
]

CLINICAL_TEXT = (
    "The patient was admitted with an acute exacerbation of chronic obstructive "
    "pulmonary disease necessitating administration of bronchodilator therapy, and "
    "the condition improved over the course of the hospitalization. "
    "Return to the emergency department immediately if you develop chest pain or "
    "difficulty breathing."
)


# --- pre-brief: every point links to a source ------------------------------

def test_prebrief_points_cite_sources():
    brief = generate_brief(NOTE_SNIPPETS, llm=MOCK)
    points = brief.points
    assert points, "brief should produce points"
    valid_ids = {s["id"] for s in NOTE_SNIPPETS}
    for p in points:
        assert p.source_ids, f"point is uncited: {p.text!r}"
        assert all(sid in valid_ids for sid in p.source_ids)
    assert brief.citation_rate == 1.0
    assert brief.uncited == []


# --- discharge: an unsupported sentence gets flagged -----------------------

def test_discharge_generation_is_grounded():
    doc = generate_discharge(ENCOUNTER, llm=MOCK)
    assert doc.all_sentences, "discharge should produce sentences"
    # The mock drafts only from the input, so nothing should be flagged.
    assert not doc.has_unsupported


def test_discharge_flags_unsupported_sentence():
    sources = [s["text"] for s in ENCOUNTER]
    source_ids = [s["id"] for s in ENCOUNTER]
    doc = DischargeDoc(sections=[
        DischargeSection(heading="Brief hospital course", sentences=[
            DischargeSentence(text="Treated with antibiotics for pneumonia.", source_ids=["E1"]),
            # Not supported by anything in the encounter — a fabricated detail.
            DischargeSentence(text="The patient underwent an emergency appendectomy."),
        ]),
    ])
    check_support(doc, sources, source_ids=source_ids)
    unsupported = [s.text for s in doc.unsupported]
    assert doc.has_unsupported
    assert "The patient underwent an emergency appendectomy." in unsupported
    assert "Treated with antibiotics for pneumonia." not in unsupported


# --- simplify: simpler text, warning kept ----------------------------------

def test_simplify_lowers_reading_level_and_keeps_warning():
    result = simplify(CLINICAL_TEXT, llm=MOCK)
    assert result.simplified
    assert result.got_simpler, (
        f"reading grade did not drop: {result.reading}")
    assert result.reading["grade_drop"] > 0
    # The emergency warning must survive and must not be flagged as dropped.
    assert not result.safety.blocked
    assert result.safety.dropped == []
    assert result.safety.source_warnings, "a warning should be detected in the source"
    assert result.safety.preservation_rate == 1.0
    assert "emergency" in result.simplified.lower()


def test_safety_check_flags_dropped_warning():
    source = "Return to the emergency department if you develop chest pain."
    # A rewrite that quietly drops the warning entirely.
    over_simplified = "You had a check-up today. Everything looks fine."
    report = safety_check(source, over_simplified)
    assert report.blocked
    assert report.dropped


# --- metrics compute --------------------------------------------------------

def test_metrics_compute():
    rl = reading_level("This is a short, simple sentence.")
    assert set(rl) == {"flesch_kincaid_grade", "flesch_reading_ease", "smog_index"}
    assert all(isinstance(v, float) for v in rl.values())

    ground = grounding_rate(
        ["Treated with antibiotics.", "The patient climbed a mountain."],
        [s["text"] for s in ENCOUNTER],
    )
    assert ground.total == 2
    assert 0.0 <= ground.grounding_rate <= 1.0
    assert ground.unsupported  # the mountain sentence is unsupported

    omit = omission_check(
        important_facts=[s["text"] for s in ENCOUNTER],
        output_text="Treated with antibiotics for pneumonia.",
    )
    assert omit.important_facts == 4
    assert 0.0 <= omit.omission_rate <= 1.0
    assert omit.missed  # several encounter facts are not in the short output
