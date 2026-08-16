"""Urgency triage: label a message emergency / urgent / routine.

Two classifiers live here, and they are meant to be compared:

- ``BaselineTriage`` is a small scikit-learn model (TF-IDF features into a
  logistic-regression head). It trains on CPU in a fraction of a second, is
  fully interpretable, and is the honest floor any fancier model has to beat.
- ``llm_triage`` asks a language model for the label through a forced tool call,
  so the answer is always one of the three classes plus a short reason.

Note on the ``emergency`` label: the deterministic gate in redflags.py is what
actually protects against a missed emergency. It runs first and above these
classifiers. An ``emergency`` label from triage is a softer, model-based signal
used for routing, never the safety backstop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .llm import BaseLLM, default_llm

URGENCY_CLASSES = ("emergency", "urgent", "routine")

# A small, hand-written seed set so the baseline trains and predicts offline
# with no downloads. data/prepare.py explains how to swap in PMR-Bench for a
# real training run.
SEED_TRAIN: List[tuple] = [
    ("I have crushing chest pain going into my arm", "emergency"),
    ("I can't breathe and my lips are turning blue", "emergency"),
    ("My face is drooping and my speech is slurred", "emergency"),
    ("I took a whole bottle of pills and I want it to end", "emergency"),
    ("There is heavy bleeding that will not stop", "emergency"),
    ("My throat is closing up and my tongue is swelling", "emergency"),
    ("I've had a high fever for three days and it's getting worse", "urgent"),
    ("The wound is red, swollen and leaking pus now", "urgent"),
    ("I can't keep any food or water down since yesterday", "urgent"),
    ("My child's rash is spreading fast and she seems very tired", "urgent"),
    ("The pain in my ear has gotten a lot worse overnight", "urgent"),
    ("I feel dizzy every time I stand up and it's not going away", "urgent"),
    ("My blood sugar readings have been high all week", "urgent"),
    ("Could I get a refill on my blood pressure prescription", "routine"),
    ("I'd like to schedule my annual check-up", "routine"),
    ("When will my lab results be ready to view", "routine"),
    ("Can you send a copy of my vaccination record", "routine"),
    ("I have a general question about my diet plan", "routine"),
    ("Please reschedule my appointment to next week", "routine"),
    ("I was wondering if I still need to fast before my blood test", "routine"),
]


@dataclass
class TriageResult:
    """One urgency verdict."""

    urgency: str
    reason: str = ""
    needs_clinician: bool = True
    method: str = "baseline"


class BaselineTriage:
    """TF-IDF + logistic-regression urgency classifier (CPU, interpretable)."""

    def __init__(self):
        self.pipeline = None

    def fit(self, texts: List[str], labels: List[str]) -> "BaselineTriage":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        self.pipeline.fit(texts, labels)
        return self

    def classify(self, message: str) -> TriageResult:
        if self.pipeline is None:
            raise RuntimeError("BaselineTriage must be fitted before use.")
        label = self.pipeline.predict([message])[0]
        proba = dict(zip(self.pipeline.classes_,
                         self.pipeline.predict_proba([message])[0]))
        top = proba.get(label, 0.0)
        return TriageResult(
            urgency=label,
            reason=f"Baseline classifier assigned '{label}' "
                   f"(confidence {top:.0%}).",
            needs_clinician=True,
            method="baseline",
        )


def default_baseline() -> BaselineTriage:
    """A baseline trained on the built-in seed set, ready to classify offline."""
    texts = [t for t, _ in SEED_TRAIN]
    labels = [y for _, y in SEED_TRAIN]
    return BaselineTriage().fit(texts, labels)


_URGENCY_TOOL = {
    "name": "classify_urgency",
    "description": "Assign an urgency class to a patient portal message.",
    "input_schema": {
        "type": "object",
        "properties": {
            "urgency": {
                "type": "string",
                "enum": list(URGENCY_CLASSES),
                "description": "How soon the message needs a clinician.",
            },
            "reason": {
                "type": "string",
                "description": "One short sentence explaining the choice.",
            },
            "needs_clinician": {
                "type": "boolean",
                "description": "Whether a clinician must review before any reply.",
            },
        },
        "required": ["urgency", "reason", "needs_clinician"],
    },
}

_TRIAGE_SYSTEM = (
    "You sort patient portal messages for a care team by how soon they need "
    "attention. You do not diagnose. Choose one of: emergency, urgent, routine. "
    "Classify conservatively: never lower the urgency just because the patient "
    "sounds calm, and when a message could be time-sensitive, pick the higher "
    "class. Always call the classify_urgency tool."
)


def llm_triage(message: str, llm: Optional[BaseLLM] = None) -> TriageResult:
    """Ask a language model for the urgency label via a forced tool call."""
    llm = llm or default_llm()
    result = llm.complete(
        system=_TRIAGE_SYSTEM,
        user=f"PATIENT MESSAGE:\n{message}",
        tool=_URGENCY_TOOL,
        max_tokens=300,
    )
    payload = result.tool_input or {}
    urgency = payload.get("urgency")
    if urgency not in URGENCY_CLASSES:
        # If the model returns nothing usable, route up rather than guess low.
        return TriageResult(
            urgency="urgent",
            reason="No clear verdict from the model; routing up for safety.",
            needs_clinician=True,
            method="llm",
        )
    return TriageResult(
        urgency=urgency,
        reason=payload.get("reason", ""),
        needs_clinician=bool(payload.get("needs_clinician", True)),
        method="llm",
    )


def triage(message: str, method: str = "baseline",
           llm: Optional[BaseLLM] = None,
           baseline: Optional[BaselineTriage] = None) -> TriageResult:
    """Classify one message. ``method`` is ``"baseline"`` or ``"llm"``."""
    if method == "llm":
        return llm_triage(message, llm=llm)
    baseline = baseline or default_baseline()
    return baseline.classify(message)
