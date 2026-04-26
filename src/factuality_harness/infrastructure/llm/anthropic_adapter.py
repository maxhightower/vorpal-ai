"""Anthropic adapter using the official ``anthropic`` SDK.

Defaults to ``claude-opus-4-7``. Supports optional adaptive thinking and
prompt caching. The harness's verification step still treats every LLM
response as untrusted text — this adapter just makes sure the call shape is
correct and idiomatic.
"""

from __future__ import annotations

import os
from typing import Any

import anthropic

from .base import LLM, LLMRequest, LLMResponse


def _split_messages(request: LLMRequest) -> tuple[str, list[dict[str, Any]]]:
    """Anthropic's API takes ``system`` separately from ``messages``."""
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for m in request.messages:
        if m.role == "system":
            system_parts.append(m.content)
        else:
            messages.append({"role": m.role, "content": m.content})
    return "\n\n".join(system_parts), messages


class AnthropicAdapter:
    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "claude-opus-4-7",
        max_tokens: int = 4096,
        client: anthropic.Anthropic | None = None,
        thinking: dict[str, Any] | None = None,
        cache_system: bool = False,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._thinking = thinking
        self._cache_system = cache_system
        if client is not None:
            self._client = client
        else:
            self._client = anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
            )

    def complete(self, request: LLMRequest) -> LLMResponse:
        system_text, messages = _split_messages(request)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": request.max_tokens or self._max_tokens,
            "messages": messages,
        }

        if system_text:
            if self._cache_system:
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                kwargs["system"] = system_text

        if self._thinking is not None:
            kwargs["thinking"] = self._thinking
        elif request.metadata.get("thinking"):
            kwargs["thinking"] = request.metadata["thinking"]

        response = self._client.messages.create(**kwargs)
        text_parts = [
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ]
        return LLMResponse(
            text="".join(text_parts),
            raw={
                "id": response.id,
                "model": response.model,
                "stop_reason": response.stop_reason,
                "usage": response.usage.model_dump() if response.usage else None,
            },
        )


_: LLM = AnthropicAdapter.__new__(AnthropicAdapter)  # protocol satisfaction without API call
