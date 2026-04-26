"""Deterministic mock LLM. Returns the last user message verbatim by default.

Used for tests and as a default in environments without API access. Real
adapters live in sibling modules (e.g. ``openai_adapter.py``) and implement the
same ``LLM`` protocol.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import LLM, LLMRequest, LLMResponse


class MockLLM:
    name = "mock"

    def __init__(self, responder: Callable[[LLMRequest], str] | None = None) -> None:
        self._responder = responder

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self._responder is not None:
            return LLMResponse(text=self._responder(request))
        # Default: echo the last user message.
        last_user = next(
            (m.content for m in reversed(request.messages) if m.role == "user"),
            "",
        )
        return LLMResponse(text=last_user)


_: LLM = MockLLM()
