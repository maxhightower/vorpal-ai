"""OpenAI adapter via raw HTTP (httpx).

Implements the ``LLM`` protocol against the OpenAI Chat Completions API
without pulling in the ``openai`` SDK. The transport is injectable so the
adapter can be unit-tested with ``httpx.MockTransport``.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import LLM, LLMRequest, LLMResponse


_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIAdapter:
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def complete(self, request: LLMRequest) -> LLMResponse:
        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set — cannot call OpenAI Chat Completions API."
            )

        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens

        response = self._client.post(
            f"{self._base_url}/chat/completions",
            json=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        return LLMResponse(
            text=choice["message"]["content"] or "",
            raw={
                "id": data.get("id"),
                "model": data.get("model"),
                "finish_reason": choice.get("finish_reason"),
                "usage": data.get("usage"),
            },
        )


_: LLM = OpenAIAdapter.__new__(OpenAIAdapter)
