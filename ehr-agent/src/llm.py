"""A small model interface with a real client and a deterministic offline mock.

The agent talks to a model through one method, ``BaseLLM.step(...)``, which
takes the running conversation plus the available tools and returns either free
text or a set of tool calls to run next. Two implementations ship here:

- ``AnthropicLLM`` wraps the official ``anthropic`` SDK. It is lazy: importing
  this module never touches the network or needs a key. The client is built the
  first time you actually call it.
- ``MockLLM`` is a deterministic stand-in. It drives the same read-then-draft
  sequence with no API key and no network, so every test and the offline demo
  run out of the box.

``default_llm()`` returns the real client when ``ANTHROPIC_API_KEY`` is set and
the mock otherwise, so identical code runs offline in CI and for real with a key.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolCall:
    """A single request from the model to run one tool."""

    id: str
    name: str
    input: dict


@dataclass
class LLMResult:
    """One model turn: free text and/or a list of tool calls to run."""

    text: str = ""
    tool_calls: list = field(default_factory=list)


class BaseLLM:
    """The one interface the agent depends on."""

    def step(self, *, system: str, messages: list, tools: list,
             max_tokens: int = 1024) -> LLMResult:
        raise NotImplementedError


# --- conversation format ----------------------------------------------------
#
# The agent keeps history in a plain, provider-neutral shape so the mock and the
# real client share it:
#
#   {"role": "user",      "content": "<text>"}
#   {"role": "assistant", "text": "<text>", "tool_calls": [ToolCall, ...]}
#   {"role": "tool",      "tool_call_id": "<id>", "name": "<tool>",
#    "content": "<json string>"}
#
# ``AnthropicLLM`` translates this into the SDK's block format on the way out.


class AnthropicLLM(BaseLLM):
    """Real client over the anthropic SDK. Lazy so import never needs a key."""

    def __init__(self, model: str = "claude-sonnet-5"):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        return self._client

    @staticmethod
    def _to_anthropic(messages: list) -> list:
        """Turn the neutral history into anthropic message blocks."""
        out: list = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks: list = []
                if m.get("text"):
                    blocks.append({"type": "text", "text": m["text"]})
                for tc in m.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": tc.id,
                                   "name": tc.name, "input": tc.input})
                out.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                block = {"type": "tool_result",
                         "tool_use_id": m["tool_call_id"], "content": m["content"]}
                # Group consecutive tool results into one user turn.
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def step(self, *, system, messages, tools, max_tokens=1024) -> LLMResult:
        r = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=self._to_anthropic(messages),
        )
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, input=dict(b.input))
            for b in r.content
            if getattr(b, "type", "") == "tool_use"
        ]
        return LLMResult(text=text, tool_calls=calls)


# --- offline stand-in -------------------------------------------------------


def _first_user_text(messages: list) -> str:
    for m in messages:
        if m["role"] == "user" and isinstance(m.get("content"), str):
            return m["content"]
    return ""


def _parse_patient_and_intent(text: str) -> tuple[str, str]:
    patient_id, intent = "", ""
    for line in text.splitlines():
        low = line.lower()
        if low.startswith("patient:"):
            patient_id = line.split(":", 1)[1].strip()
        elif low.startswith("intent:"):
            intent = line.split(":", 1)[1].strip()
    return patient_id, intent


def _tools_called(messages: list) -> list:
    return [tc.name for m in messages if m["role"] == "assistant"
            for tc in m.get("tool_calls", [])]


def _read_results(messages: list, tool_name: str) -> Optional[list]:
    """Return the parsed output of the most recent call to ``tool_name``."""
    for m in reversed(messages):
        if m["role"] == "tool" and m.get("name") == tool_name:
            try:
                return json.loads(m["content"])
            except (ValueError, TypeError):
                return None
    return None


def _draft_from_reads(messages: list, patient_id: str, intent: str) -> dict:
    """Build a grounded draft order from whatever the agent read.

    The mock does not reason clinically. It picks the first out-of-range lab it
    saw and drafts a follow-up recheck, citing that observation as evidence, so
    the offline path always produces a justified, traceable proposal. If no lab
    is abnormal it drafts a routine follow-up keyed to a recorded condition.
    """
    labs = _read_results(messages, "get_labs") or []
    conditions = _read_results(messages, "get_conditions") or []

    abnormal = next((o for o in labs if o.get("interpretation", "N") != "N"), None)
    if abnormal:
        return {
            "resourceType": "ServiceRequest",
            "patient_id": patient_id,
            "code_system": "http://loinc.org",
            "code": abnormal.get("code", ""),
            "display": f"Recheck {abnormal.get('display', 'lab')}",
            "priority": "routine",
            "reason_text": (
                f"Follow-up requested ('{intent}'). Most recent "
                f"{abnormal.get('display', 'lab')} was "
                f"{abnormal.get('value', '?')} {abnormal.get('unit', '')} "
                f"(flag {abnormal.get('interpretation')}); a recheck confirms "
                f"the trend before any change to therapy."
            ),
            "evidence": [abnormal.get("reference", "")],
        }

    cond = conditions[0] if conditions else {}
    return {
        "resourceType": "ServiceRequest",
        "patient_id": patient_id,
        "code_system": "http://loinc.org",
        "code": "24323-8",
        "display": "Comprehensive metabolic panel",
        "priority": "routine",
        "reason_text": (
            f"Follow-up requested ('{intent}'). Routine monitoring for "
            f"{cond.get('display', 'the recorded problem list')}."
        ),
        "evidence": [cond.get("reference", "")] if cond else [],
    }


class MockLLM(BaseLLM):
    """Deterministic offline stand-in. No key, no network, stable output.

    It walks a fixed plan: read conditions, then medications, then labs, then
    draft one order, then stop. Each decision is a function of what has already
    been called, so the same input always yields the same trajectory.
    """

    def step(self, *, system, messages, tools, max_tokens=1024) -> LLMResult:
        patient_id, intent = _parse_patient_and_intent(_first_user_text(messages))
        called = _tools_called(messages)

        if "get_conditions" not in called:
            return LLMResult(tool_calls=[ToolCall(
                "c1", "get_conditions", {"patient_id": patient_id})])
        if "get_medications" not in called:
            return LLMResult(tool_calls=[ToolCall(
                "m1", "get_medications", {"patient_id": patient_id})])
        if "get_labs" not in called:
            return LLMResult(tool_calls=[ToolCall(
                "l1", "get_labs", {"patient_id": patient_id})])
        if "propose_order" not in called:
            draft_input = _draft_from_reads(messages, patient_id, intent)
            return LLMResult(tool_calls=[ToolCall("p1", "propose_order", draft_input)])

        return LLMResult(text=(
            "I read the patient's conditions, active medications, and recent "
            "labs, then drafted one order for review. Nothing has been written "
            "to the record. Please approve or reject the proposal."))


def default_llm() -> BaseLLM:
    """Real client when a key is present, else the offline mock."""
    return AnthropicLLM() if os.environ.get("ANTHROPIC_API_KEY") else MockLLM()
