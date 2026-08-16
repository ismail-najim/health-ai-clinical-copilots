"""documentation — pre-visit briefs, discharge summaries, and plain-English rewrites.

An assistive drafting tool for clinical paperwork. It drafts; a clinician
reviews and signs. Every line is either grounded in the input or flagged.
"""

from .llm import (
    AnthropicLLM,
    BaseLLM,
    LLMResult,
    MockLLM,
    default_llm,
)
from .prebrief import Brief, generate_brief
from .discharge import DischargeDoc, check_support, generate_discharge
from .simplify import SafetyReport, SimplifyResult, safety_check, simplify
from .eval import (
    grounding_rate,
    omission_check,
    reading_level,
    reading_level_change,
    support_scores,
)

__all__ = [
    "BaseLLM",
    "LLMResult",
    "AnthropicLLM",
    "MockLLM",
    "default_llm",
    "generate_brief",
    "Brief",
    "generate_discharge",
    "DischargeDoc",
    "check_support",
    "simplify",
    "SimplifyResult",
    "SafetyReport",
    "safety_check",
    "reading_level",
    "reading_level_change",
    "grounding_rate",
    "omission_check",
    "support_scores",
]
