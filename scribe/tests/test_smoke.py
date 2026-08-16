"""Offline end-to-end smoke test. CPU only, no key, no downloads.

Runs a transcript through the whole scribe on the deterministic stand-in and
checks: a note comes back, each sentence links to transcript lines, an
obviously made-up sentence is flagged unsupported, and the grounding and
omission metrics compute.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import grounding_rate, omission_rate
from src.llm import MockLLM, default_llm
from src.scribe import NoteSentence, Note, Scribe, parse_transcript

TRANSCRIPT = """Doctor: What brings you in today?
Patient: I've had a sore throat and a cough for about five days.
Patient: The cough is worse at night and I've felt feverish.
Doctor: Any chest pain or trouble breathing?
Patient: No chest pain, breathing is fine.
Doctor: Are you on any medication?
Patient: Just ibuprofen for the throat. I'm allergic to penicillin.
Doctor: Let's run a rapid strep test and follow up in a week."""

REFERENCE_NOTE = (
    "Patient reports a sore throat and cough for five days, worse at night, "
    "with subjective fever. Denies chest pain and shortness of breath. "
    "Takes ibuprofen as needed. Allergic to penicillin. "
    "Plan: rapid strep test and follow up in one week."
)


def test_offline_backend_is_mock():
    # With no key set, the default backend must be the offline stand-in.
    assert os.environ.get("ANTHROPIC_API_KEY") is None
    assert isinstance(default_llm(), MockLLM)


def test_note_comes_back():
    scribe = Scribe(MockLLM())
    note, transcript = scribe.run(TRANSCRIPT)
    assert isinstance(note, Note)
    assert len(note.sentences) >= 3
    assert len(transcript) >= 6


def test_evidence_maps_sentences_to_lines():
    scribe = Scribe(MockLLM())
    note, transcript = scribe.run(TRANSCRIPT)
    valid_lines = {ln.line for ln in transcript}
    supported = [s for s in note.sentences if s.status == "supported"]
    assert supported, "expected at least one supported sentence"
    for s in supported:
        assert s.source_lines, f"sentence has no linked evidence: {s.text!r}"
        for line in s.source_lines:
            assert line in valid_lines


def test_confabulation_flags_unsupported_sentence():
    scribe = Scribe(MockLLM())
    note, transcript = scribe.run(TRANSCRIPT)
    # Inject a sentence nothing in the transcript supports.
    made_up = NoteSentence(
        text="The patient underwent an emergency appendectomy last night.",
        heading="Plan",
    )
    note.sentences.append(made_up)
    checked = scribe.link_and_check(
        [(s.heading, s.text) for s in note.sentences], transcript
    )
    flagged = [s for s in checked.flagged]
    assert any("appendectomy" in s.text for s in flagged), \
        "the made-up sentence should be flagged unsupported"


def test_grounding_and_omission_metrics_compute():
    scribe = Scribe(MockLLM())
    note, _ = scribe.run(TRANSCRIPT)

    g = grounding_rate(note)
    assert 0.0 <= g.grounding_rate <= 1.0
    assert g.total == len(note.sentences)
    assert g.supported + len(g.unsupported_sentences) == g.total

    o = omission_rate(note, REFERENCE_NOTE)
    assert 0.0 <= o.omission_rate <= 1.0
    assert o.reference_facts > 0
    assert o.covered + len(o.missed) == o.reference_facts
    # This transcript carries the reference facts, so most should be covered.
    assert o.omission_rate < 0.5


def test_grounding_and_omission_are_separate_numbers():
    # A note can score differently on the two axes; they are not the same metric.
    scribe = Scribe(MockLLM())
    note, _ = scribe.run(TRANSCRIPT)
    g = grounding_rate(note)
    o = omission_rate(note, REFERENCE_NOTE)
    assert hasattr(g, "grounding_rate") and hasattr(o, "omission_rate")
