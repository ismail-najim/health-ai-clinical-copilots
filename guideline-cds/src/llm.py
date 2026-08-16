"""LLM client with an injectable, deterministic offline stand-in.

The decision-support flow calls a model through a single `.complete(...)` method.
`AnthropicLLM` wraps the official SDK for real runs; `MockLLM` returns a
deterministic, grounded answer built only from the retrieved snippets, so the whole
tool (and its test suite) runs offline with no API key. `default_llm()` picks the
real client when `ANTHROPIC_API_KEY` is set, otherwise the mock.

The mock is intentionally simple: it stitches together the supplied context
snippets and tags each sentence with the snippet id it came from. That mirrors the
contract the real model is asked to follow (every clinical sentence ends with its
source id) so the citation gate behaves the same offline and online.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResult:
    text: str = ""


class BaseLLM:
    def complete(self, *, system: str = "", user: str = "",
                 max_tokens: int = 700) -> LLMResult:
        raise NotImplementedError


class AnthropicLLM(BaseLLM):
    """Real client. Construction is lazy so importing this module never fails
    when no key is present."""

    def __init__(self, model: str = "claude-sonnet-5"):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic()  # reads ANTHROPIC_API_KEY
        return self._client

    def complete(self, *, system: str = "", user: str = "",
                 max_tokens: int = 700) -> LLMResult:
        r = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return LLMResult(text=text.strip())


# Snippets are passed to the model inside a block of lines like "[S1] text...".
_CTX_LINE = re.compile(r"^\[([A-Za-z0-9]+)\]\s*(.+)$")


def _parse_context(user: str) -> list[tuple[str, str]]:
    """Pull the "[id] text" context lines back out of the user prompt."""
    out = []
    for line in user.splitlines():
        m = _CTX_LINE.match(line.strip())
        if m:
            out.append((m.group(1), m.group(2).strip()))
    return out


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0].strip() if parts else text.strip()


class MockLLM(BaseLLM):
    """Deterministic offline stand-in.

    Answers strictly from the context lines embedded in the prompt: it takes the
    first sentence of each retrieved snippet and appends that snippet's citation id.
    If the prompt carries no context, it returns the abstain sentinel. To exercise
    the citation gate, it also appends one deliberately uncited sentence.
    """

    ABSTAIN = "INSUFFICIENT_EVIDENCE"

    def complete(self, *, system: str = "", user: str = "",
                 max_tokens: int = 700) -> LLMResult:
        ctx = _parse_context(user)
        if not ctx:
            return LLMResult(text=self.ABSTAIN)
        sentences = []
        for cid, text in ctx:
            body = _first_sentence(text).rstrip(".")
            sentences.append(f"{body} [{cid}].")
        # One extra, deliberately unsupported line so the gate has something to drop.
        sentences.append("Overall this is a reasonable general approach.")
        return LLMResult(text=" ".join(sentences))


def default_llm() -> BaseLLM:
    """Real client if a key is present, else the offline mock."""
    return AnthropicLLM() if os.environ.get("ANTHROPIC_API_KEY") else MockLLM()
