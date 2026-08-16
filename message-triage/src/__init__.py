"""message-triage: sort patient messages by urgency and draft replies.

A patient message flows through a fixed pipeline:

    red-flag gate  ->  urgency triage  ->  condense the question  ->  draft reply

The red-flag gate runs first and is a plain rule, not a model, so an emergency
message can never slip through to a routine reply. Every draft is held for a
clinician to approve; nothing is sent automatically.
"""

__all__ = ["llm", "redflags", "triage", "summarize", "reply", "eval"]
