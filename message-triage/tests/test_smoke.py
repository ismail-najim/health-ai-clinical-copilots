"""Offline smoke tests. CPU only, no network, no downloads.

Every test uses the deterministic stand-in from src/llm.py, so results are
stable and no API key is needed. Run with: python -m pytest tests/ -q
"""
from __future__ import annotations

from src.eval import over_reassures, red_flag_recall
from src.llm import MockLLM
from src.redflags import screen
from src.reply import approve, draft_reply
from src.summarize import summarize
from src.triage import default_baseline, llm_triage


MOCK = MockLLM()


def test_red_flag_gate_catches_emergency_and_blocks_routine_reply():
    """An emergency message trips the gate, and the pipeline stops there."""
    message = "I feel like an elephant is sitting on my chest and my arm is numb"
    flag = screen(message)
    assert flag.is_emergency
    assert flag.blocks_routine_reply
    assert flag.category is not None
    assert "emergency" in flag.guidance.lower()
    # A caller respecting the gate never drafts a routine reply for this message.
    # We assert the gate itself is what would prevent it.
    assert flag.blocks_routine_reply is True


def test_benign_message_gets_urgency_label_and_draft():
    """A routine message passes the gate, gets a label and a draft reply."""
    message = "Could I get a refill on my blood pressure prescription please?"
    assert not screen(message).is_emergency

    # Baseline (scikit-learn) produces a valid label.
    base = default_baseline().classify(message)
    assert base.urgency in ("emergency", "urgent", "routine")

    # LLM option also produces a valid label.
    verdict = llm_triage(message, llm=MOCK)
    assert verdict.urgency in ("emergency", "urgent", "routine")

    # A draft is produced.
    summary = summarize(message, llm=MOCK)
    drafted = draft_reply(summary.question, message, record=["On lisinopril 10mg"],
                          llm=MOCK)
    assert drafted.body.strip()


def test_summarizer_condenses():
    """A rambling message is reduced to a shorter, single question."""
    message = (
        "Hi, I hope you're well. I've been meaning to write for a while now. "
        "I was at my daughter's wedding last month which was lovely. Anyway, "
        "what I actually wanted to ask is whether I should keep taking my "
        "vitamin D through the summer?"
    )
    summary = summarize(message, llm=MOCK)
    assert summary.question.strip()
    assert len(summary.question) < len(message)


def test_draft_is_never_auto_sent():
    """The approval gate holds: a fresh draft is not sent until approved."""
    drafted = draft_reply("Should I keep taking vitamin D?", "some message",
                          llm=MOCK)
    assert drafted.sent is False
    assert drafted.approved is False
    assert drafted.requires_approval is True

    # An empty clinician name cannot send.
    try:
        approve(drafted, "")
        assert False, "empty approval should have raised"
    except ValueError:
        pass
    assert drafted.sent is False

    # A named clinician is the only way it sends.
    approve(drafted, "Dr. Rivera")
    assert drafted.sent is True
    assert drafted.approved_by == "Dr. Rivera"


def test_red_flag_recall_metric_computes():
    """The headline metric runs offline and catches every seeded emergency."""
    report = red_flag_recall()
    assert report.total_emergencies > 0
    assert report.recall == 1.0
    assert report.false_negatives == []
    assert report.near_perfect


def test_over_reassurance_detector():
    """A soothe-without-care draft is flagged; a careful one is not."""
    bad = "It's probably nothing to worry about, no need to come in."
    good = ("This may be minor, but if it gets worse please contact the care "
            "team or seek urgent care.")
    assert over_reassures(bad) is True
    assert over_reassures(good) is False
