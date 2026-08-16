"""Offline smoke test — CPU, no network, no downloads, no API key.

Uses the deterministic MockLLM and the built-in example corpus. Covers:
  - retrieval returns a relevant snippet for an in-corpus question
  - a normal question yields a cited, non-abstaining answer
  - a question with no supporting snippet triggers the abstain path
  - the citation gate drops sentences whose citations are not in context
  - the attribution metric computes
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cds import GuidelineCDS, citation_gate, ABSTAIN_MESSAGE
from src.eval import statement_attribution_rate
from src.llm import MockLLM
from src.retriever import Retriever
from src.tools import uspstf_lookup, openfda_label_lookup


def _copilot():
    # Force the offline mock so the test never needs a key or the network.
    return GuidelineCDS(llm=MockLLM(), retriever=Retriever())


def test_retrieval_returns_relevant_snippet():
    r = Retriever()
    hits = r.retrieve("first-line treatment for type 2 diabetes", k=3)
    assert hits, "expected at least one hit for an in-corpus question"
    assert hits[0]["id"] == "G2", f"expected diabetes snippet first, got {hits[0]['id']}"
    assert hits[0]["score"] > 0


def test_normal_question_yields_cited_answer():
    result = _copilot().answer("first-line management of uncomplicated hypertension")
    assert not result.abstained
    assert result.answer and result.answer != ABSTAIN_MESSAGE
    assert result.sources, "a non-abstaining answer must carry sources"
    # every listed source id actually appears in the answer text
    for s in result.sources:
        assert f"[{s['id']}]" in result.answer


def test_out_of_corpus_question_abstains():
    result = _copilot().answer("what is the best programming language for web design")
    assert result.abstained
    assert result.answer == ABSTAIN_MESSAGE


def test_citation_gate_drops_unsupported_sentences():
    valid = {"G1"}
    text = ("Thiazide diuretics are a first-line option [G1]. "
            "Some other unproven claim with a bogus source [Z9]. "
            "A confident sentence with no citation at all.")
    kept, dropped = citation_gate(text, valid)
    kept_text = " ".join(kept)
    assert "[G1]" in kept_text
    assert "[Z9]" not in kept_text
    reasons = {d["reason"] for d in dropped}
    assert "citation not in retrieved context" in reasons
    assert "no citation" in reasons
    assert len(dropped) == 2


def test_gate_all_unsupported_forces_abstain():
    # A mock that emits only uncited clinical claims.
    class NoCiteLLM(MockLLM):
        def complete(self, *, system="", user="", max_tokens=700):
            from src.llm import LLMResult
            return LLMResult(text="This is a confident clinical claim with no source at all.")

    result = GuidelineCDS(llm=NoCiteLLM(), retriever=Retriever()).answer(
        "first-line management of uncomplicated hypertension")
    assert result.abstained
    assert result.answer == ABSTAIN_MESSAGE
    assert result.dropped, "the uncited sentence should be recorded as dropped"


def test_attribution_metric_computes():
    result = _copilot().answer("first-line pharmacotherapy for type 2 diabetes")
    valid_ids = {h["id"] for h in result.retrieved}
    metric = statement_attribution_rate(result.answer, valid_ids)
    assert metric["rate"] is not None
    # after the gate, every surviving clinical sentence is cited -> full rate
    assert metric["rate"] == 1.0
    assert metric["total"] >= 1


def test_tools_offline_fallback():
    recs = uspstf_lookup(keyword="lung", offline=True)
    assert recs and all("url" in r and "id" in r for r in recs)
    label = openfda_label_lookup("metformin", offline=True)
    assert label and label[0]["section"] == "boxed_warning"
