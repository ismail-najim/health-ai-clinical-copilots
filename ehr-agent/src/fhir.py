"""A tiny in-memory patient store shaped like FHIR, plus notes on a real server.

The offline demo and the whole test suite run against ``InMemoryStore``: a
handful of synthetic patients with conditions, medications, and lab results,
held in plain dictionaries. No network, no database, no keys.

Two things matter about the store's design:

- The read functions (``search_patients``, ``get_conditions``,
  ``get_medications``, ``get_labs``) are the only surface the agent's tools use.
- Writing is separate and deliberate. ``create_from_proposal`` is the single
  place a draft becomes a stored record, and only the human-approval path in the
  demo ever calls it. The agent has no route to it. The store also counts writes
  so a test can prove the agent never triggered one.

Pointing this at a real FHIR server
------------------------------------
The synthetic records here mirror FHIR R4 resource fields. To run against a real
server (for example a self-hosted HAPI FHIR container, or a public R4 test
server such as ``https://hapi.fhir.org/baseR4``), keep the same read/write split
and swap the store body for HTTP calls with ``requests``:

    import os, requests
    BASE = os.environ.get("FHIR_BASE", "http://localhost:8080/fhir")
    HDR = {"Accept": "application/fhir+json", "Content-Type": "application/fhir+json"}

    def get_labs(patient_id):
        r = requests.get(f"{BASE}/Observation",
                         params={"patient": patient_id, "category": "laboratory",
                                 "_sort": "-date", "_count": "20"},
                         headers=HDR, timeout=30)
        r.raise_for_status()
        return _flatten_observations(r.json())  # pull code/value/interpretation out of the Bundle

Load synthetic patients with Synthea (Apache-2.0), which exports FHIR R4 bundles
you POST into the server. The important rule stays the same everywhere: the
agent only reads and drafts; the write lives behind a human approval.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InMemoryStore:
    """A fake FHIR patient store. Reads are open; the one write path is counted."""

    patients: dict = field(default_factory=dict)
    conditions: dict = field(default_factory=dict)
    medications: dict = field(default_factory=dict)
    labs: dict = field(default_factory=dict)
    #: Records committed through the approval path. The agent never touches this.
    committed: list = field(default_factory=list)

    # --- read surface (what the agent's tools call) ---

    def search_patients(self, query: str = "") -> list:
        q = query.lower().strip()
        out = []
        for pid, p in self.patients.items():
            if not q or q in p["name"].lower() or q == pid:
                out.append({"reference": f"Patient/{pid}", "id": pid, **p})
        return out

    def read_patient(self, patient_id: str) -> dict:
        p = self.patients.get(patient_id)
        return {"reference": f"Patient/{patient_id}", "id": patient_id, **p} if p else {}

    def get_conditions(self, patient_id: str) -> list:
        return list(self.conditions.get(patient_id, []))

    def get_medications(self, patient_id: str) -> list:
        return list(self.medications.get(patient_id, []))

    def get_labs(self, patient_id: str) -> list:
        return list(self.labs.get(patient_id, []))

    # --- write path (human approval only) ---

    def create_from_proposal(self, proposal: dict) -> dict:
        """Commit an approved draft. This is the ONLY write, and the demo's
        Approve button is the only caller. The agent has no reference to it."""
        record = dict(proposal)
        record["status"] = "active"
        record["id"] = f"order-{len(self.committed) + 1}"
        self.committed.append(record)
        return {"id": record["id"], "status": record["status"],
                "resourceType": record.get("resourceType")}

    @property
    def write_count(self) -> int:
        return len(self.committed)


def default_store() -> InMemoryStore:
    """A small, deterministic synthetic cohort for the demo and tests.

    The records are invented and abbreviated; codes are illustrative. They are
    enough to exercise the read-then-draft flow without any real patient data.
    """
    store = InMemoryStore()

    store.patients = {
        "patient-1": {"name": "Avery Stone", "gender": "female", "age": 58},
        "patient-2": {"name": "Jordan Reyes", "gender": "male", "age": 47},
        "patient-3": {"name": "Priya Malik", "gender": "female", "age": 63},
    }

    store.conditions = {
        "patient-1": [
            {"reference": "Condition/c-1a", "code": "55822004",
             "display": "Hyperlipidemia", "onset": "2019-04-12", "status": "active"},
            {"reference": "Condition/c-1b", "code": "38341003",
             "display": "Essential hypertension", "onset": "2017-08-01",
             "status": "active"},
        ],
        "patient-2": [
            {"reference": "Condition/c-2a", "code": "44054006",
             "display": "Type 2 diabetes mellitus", "onset": "2021-01-20",
             "status": "active"},
        ],
        "patient-3": [
            {"reference": "Condition/c-3a", "code": "195967001",
             "display": "Asthma", "onset": "2005-06-30", "status": "active"},
        ],
    }

    store.medications = {
        "patient-1": [
            {"reference": "MedicationRequest/m-1a", "code": "197361",
             "display": "Amlodipine 5 mg oral tablet", "status": "active"},
        ],
        "patient-2": [
            {"reference": "MedicationRequest/m-2a", "code": "860975",
             "display": "Metformin 500 mg oral tablet", "status": "active"},
        ],
        "patient-3": [
            {"reference": "MedicationRequest/m-3a", "code": "745752",
             "display": "Albuterol 90 mcg/actuation inhaler", "status": "active"},
        ],
    }

    # Labs carry a FHIR-style interpretation flag: "N" normal, "H" high,
    # "L" low, "HH"/"LL" critical. The agent's draft cites these by reference.
    store.labs = {
        "patient-1": [
            {"reference": "Observation/o-1a", "code": "18262-6",
             "display": "LDL cholesterol", "value": 168, "unit": "mg/dL",
             "date": "2026-07-30", "interpretation": "H"},
            {"reference": "Observation/o-1b", "code": "2093-3",
             "display": "Total cholesterol", "value": 244, "unit": "mg/dL",
             "date": "2026-07-30", "interpretation": "H"},
            {"reference": "Observation/o-1c", "code": "2085-9",
             "display": "HDL cholesterol", "value": 52, "unit": "mg/dL",
             "date": "2026-07-30", "interpretation": "N"},
        ],
        "patient-2": [
            {"reference": "Observation/o-2a", "code": "4548-4",
             "display": "Hemoglobin A1c", "value": 9.1, "unit": "%",
             "date": "2026-08-02", "interpretation": "H"},
            {"reference": "Observation/o-2b", "code": "2339-0",
             "display": "Glucose", "value": 178, "unit": "mg/dL",
             "date": "2026-08-02", "interpretation": "H"},
        ],
        "patient-3": [
            {"reference": "Observation/o-3a", "code": "718-7",
             "display": "Hemoglobin", "value": 13.6, "unit": "g/dL",
             "date": "2026-05-15", "interpretation": "N"},
        ],
    }

    return store
