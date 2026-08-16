"""The agent's tools: read the record freely, and draft an order (never write).

Four read tools expose the store, and one draft tool (``propose_order``) turns
the agent's intent into a structured draft resource. The draft tool is the whole
point of the safety story: it returns a proposal object with ``status: "draft"``
and does not — cannot — commit anything. There is no write function anywhere in
this module. Committing an approved draft happens elsewhere, on a human's click.

Tool schemas are in the anthropic tool format so the real client can pass them
straight through. The offline mock reads the same names and shapes.
"""
from __future__ import annotations

from typing import Callable

from .fhir import InMemoryStore

# --- tool schemas -----------------------------------------------------------

PATIENT_SEARCH = {
    "name": "patient_search",
    "description": "Find patients by name or id. Read-only. Returns a list.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string",
                                 "description": "Name fragment or patient id."}},
        "required": [],
    },
}

GET_CONDITIONS = {
    "name": "get_conditions",
    "description": "List a patient's problem-list conditions. Read-only.",
    "input_schema": {
        "type": "object",
        "properties": {"patient_id": {"type": "string"}},
        "required": ["patient_id"],
    },
}

GET_MEDICATIONS = {
    "name": "get_medications",
    "description": "List a patient's active medications. Read-only.",
    "input_schema": {
        "type": "object",
        "properties": {"patient_id": {"type": "string"}},
        "required": ["patient_id"],
    },
}

GET_LABS = {
    "name": "get_labs",
    "description": ("List a patient's recent lab observations with values and "
                    "interpretation flags. Read-only."),
    "input_schema": {
        "type": "object",
        "properties": {"patient_id": {"type": "string"}},
        "required": ["patient_id"],
    },
}

PROPOSE_ORDER = {
    "name": "propose_order",
    "description": (
        "Draft a lab/imaging order (ServiceRequest) or a medication order "
        "(MedicationRequest) for a clinician to review. This does NOT write to "
        "the record. It returns a draft that a human must approve before it "
        "becomes real. Cite the exact record references that justify the order."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "resourceType": {"type": "string",
                             "enum": ["ServiceRequest", "MedicationRequest"]},
            "patient_id": {"type": "string"},
            "code_system": {"type": "string",
                            "description": "e.g. http://loinc.org or RxNorm"},
            "code": {"type": "string"},
            "display": {"type": "string"},
            "priority": {"type": "string",
                         "enum": ["routine", "urgent", "asap", "stat"]},
            "reason_text": {"type": "string",
                            "description": "Plain-language clinical rationale."},
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Record references (e.g. 'Observation/o-1a') "
                               "that justify this order.",
            },
        },
        "required": ["resourceType", "patient_id", "code_system", "code",
                     "display", "priority", "reason_text", "evidence"],
    },
}

#: Everything the agent may call. ``propose_order`` is the only non-read tool,
#: and it produces a draft, not a write.
TOOL_SCHEMAS = [PATIENT_SEARCH, GET_CONDITIONS, GET_MEDICATIONS, GET_LABS,
                PROPOSE_ORDER]

#: Read tools resolve to a store call. Note there is no write tool here.
READ_TOOLS = {"patient_search", "get_conditions", "get_medications", "get_labs"}
DRAFT_TOOLS = {"propose_order"}


def read_dispatch(store: InMemoryStore) -> dict:
    """Map read-tool names to store functions for the given store."""
    return {
        "patient_search": lambda a: store.search_patients(a.get("query", "")),
        "get_conditions": lambda a: store.get_conditions(a["patient_id"]),
        "get_medications": lambda a: store.get_medications(a["patient_id"]),
        "get_labs": lambda a: store.get_labs(a["patient_id"]),
    }


def run_read_tool(store: InMemoryStore, name: str, args: dict):
    fn: Callable = read_dispatch(store).get(name)
    if fn is None:
        raise ValueError(f"unknown read tool: {name}")
    return fn(args)


def build_draft(args: dict) -> dict:
    """Assemble a FHIR-shaped draft from ``propose_order`` arguments.

    No write happens here. The returned resource carries ``status: "draft"`` and
    stays a draft until a human approves it through the store's write path.
    """
    coding = [{"system": args["code_system"], "code": args["code"],
               "display": args["display"]}]
    draft = {
        "resourceType": args["resourceType"],
        "status": "draft",  # never "active" until a human approves
        "intent": "order",
        "priority": args["priority"],
        "subject": {"reference": f"Patient/{args['patient_id']}"},
        "reasonCode": [{"text": args["reason_text"]}],
        "supportingInfo": [{"reference": e} for e in args["evidence"] if e],
    }
    if args["resourceType"] == "ServiceRequest":
        draft["code"] = {"coding": coding}
    else:  # MedicationRequest
        draft["medicationCodeableConcept"] = {"coding": coding}
    return draft
