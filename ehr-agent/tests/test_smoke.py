"""Offline smoke tests. CPU-only, no network, no API key — the mock drives them.

These lock in the behavior the whole tool promises: the agent reads the record,
drafts an order, never writes on its own, stays inside its step budget, and every
draft is justified.
"""
from src.agent import run
from src.eval import run_checks
from src.fhir import default_store
from src.llm import MockLLM


def test_agent_reads_then_drafts():
    store = default_store()
    result = run("patient-1", "Follow up on the abnormal lipid panel.",
                 llm=MockLLM(), store=store)
    # It read the chart...
    read_tools = {r["tool"] for r in result["reads"]}
    assert {"get_conditions", "get_medications", "get_labs"} <= read_tools
    # ...and produced at least one proposal.
    assert len(result["proposals"]) >= 1


def test_write_gate_holds_nothing_auto_written():
    store = default_store()
    result = run("patient-1", "Follow up on the abnormal lipid panel.",
                 llm=MockLLM(), store=store)
    assert store.write_count == 0
    assert result["write_count"] == 0
    for p in result["proposals"]:
        assert p["status"] == "draft"


def test_proposals_carry_rationale_and_evidence():
    store = default_store()
    result = run("patient-2", "Recheck diabetes control.",
                 llm=MockLLM(), store=store)
    assert result["proposals"]
    for p in result["proposals"]:
        assert p["reasonCode"][0]["text"].strip()
        assert any(s.get("reference") for s in p["supportingInfo"])


def test_loop_is_bounded():
    store = default_store()
    result = run("patient-1", "Follow up.", llm=MockLLM(), store=store,
                 max_steps=6)
    assert result["steps"] <= result["max_steps"]


def test_eval_checks_pass_offline():
    report = run_checks()
    assert report["passed"], report["checks"]


def test_approval_is_the_only_write_path():
    # The agent leaves the store untouched; a human approval commits the draft.
    store = default_store()
    result = run("patient-1", "Follow up on the abnormal lipid panel.",
                 llm=MockLLM(), store=store)
    assert store.write_count == 0
    committed = store.create_from_proposal(result["proposals"][0])
    assert store.write_count == 1
    assert committed["status"] == "active"
