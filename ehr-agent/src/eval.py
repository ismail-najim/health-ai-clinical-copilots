"""Checks that hold the safety promise: no auto-writes, and justified drafts.

These are deliberately simple and deterministic. They run offline against the
mock and confirm the two invariants the whole tool rests on:

1. The agent never writes to the record on its own.
2. Every proposed order is a draft and carries both a rationale and the record
   evidence that justifies it, and the loop stays within its step budget.

``run_checks(...)`` runs the agent and returns a report you can print or assert
on. Each check returns ``(passed: bool, message: str)``.
"""
from __future__ import annotations

from .agent import run
from .fhir import InMemoryStore, default_store
from .llm import BaseLLM, MockLLM


def check_no_autowrite(store: InMemoryStore, result: dict) -> tuple:
    """The store must be untouched after the agent runs."""
    n = store.write_count
    if n != 0 or result.get("write_count", 0) != 0:
        return False, f"SAFETY VIOLATION: {n} record write(s) from the agent path"
    return True, "no auto-writes: the agent proposed only, wrote nothing"


def check_proposals_grounded(result: dict) -> tuple:
    """Every proposal is a draft with a rationale and cited evidence."""
    proposals = result.get("proposals", [])
    if not proposals:
        return False, "no proposals were produced"
    for i, p in enumerate(proposals):
        if p.get("status") != "draft":
            return False, f"proposal {i} is not a draft (status={p.get('status')})"
        rationale = (p.get("reasonCode") or [{}])[0].get("text", "").strip()
        if not rationale:
            return False, f"proposal {i} has no rationale"
        evidence = [e for e in p.get("supportingInfo", []) if e.get("reference")]
        if not evidence:
            return False, f"proposal {i} cites no record evidence"
    return True, f"{len(proposals)} draft(s), each with a rationale and evidence"


def check_bounded(result: dict) -> tuple:
    """The loop finished within its step budget."""
    steps, cap = result.get("steps", 0), result.get("max_steps", 0)
    if steps > cap:
        return False, f"loop exceeded its budget ({steps} > {cap})"
    return True, f"loop bounded: {steps} step(s), budget {cap}"


def run_checks(patient_id: str = "patient-1",
               intent: str = "Follow up on the abnormal lipid panel.",
               *, llm: BaseLLM = None, store: InMemoryStore = None) -> dict:
    """Run the agent offline and evaluate the invariants."""
    llm = llm or MockLLM()
    store = store or default_store()
    result = run(patient_id, intent, llm=llm, store=store)

    checks = {
        "no_autowrite": check_no_autowrite(store, result),
        "proposals_grounded": check_proposals_grounded(result),
        "bounded_loop": check_bounded(result),
    }
    passed = all(ok for ok, _ in checks.values())
    return {"passed": passed, "checks": checks, "result": result}


if __name__ == "__main__":
    report = run_checks()
    for name, (ok, msg) in report["checks"].items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
    print("\nALL CHECKS PASSED" if report["passed"] else "\nCHECKS FAILED")
