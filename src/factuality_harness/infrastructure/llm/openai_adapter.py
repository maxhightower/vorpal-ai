"""OpenAI adapter (deferred).

The MVP does not require an LLM provider. This file documents the contract: a
real adapter would call the OpenAI Chat Completions API and return an
``LLMResponse``. Keep all vendor-specific logic confined here so swapping
providers does not ripple through the application layer.
"""

from __future__ import annotations

from .base import LLM, LLMRequest, LLMResponse


class OpenAIAdapter:
    name = "openai"

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        self._api_key = api_key
        self._model = model

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError(
            "OpenAIAdapter is intentionally unimplemented in the MVP. "
            "Implement the HTTP call here or supply your own LLM adapter."
        )


_: LLM = OpenAIAdapter()
