"""OpenAI adapter test using ``httpx.MockTransport`` — no network calls."""

from __future__ import annotations

import json

import httpx
import pytest

from factuality_harness.infrastructure.llm.base import LLMMessage, LLMRequest
from factuality_harness.infrastructure.llm.openai_adapter import OpenAIAdapter


def _make_adapter(handler):
    transport = httpx.MockTransport(handler)
    return OpenAIAdapter(api_key="sk-test", transport=transport)


def test_request_shape_and_response_parsing():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "the answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )

    adapter = _make_adapter(handler)
    response = adapter.complete(
        LLMRequest(
            messages=[
                LLMMessage(role="user", content="hi"),
            ],
            max_tokens=64,
        )
    )

    assert response.text == "the answer"
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["authorization"] == "Bearer sk-test"
    body = captured["body"]
    assert body["model"] == "gpt-4o-mini"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["max_tokens"] == 64


def test_raises_when_api_key_missing():
    adapter = OpenAIAdapter(api_key="", transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        adapter.complete(LLMRequest(messages=[LLMMessage(role="user", content="x")]))


def test_propagates_http_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    adapter = _make_adapter(handler)
    with pytest.raises(httpx.HTTPStatusError):
        adapter.complete(LLMRequest(messages=[LLMMessage(role="user", content="x")]))
