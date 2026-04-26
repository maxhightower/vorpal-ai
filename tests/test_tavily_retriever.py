from __future__ import annotations

import json

import httpx

from factuality_harness.infrastructure.retrieval.tavily_web_retriever import (
    TavilyWebRetriever,
)


def _make_retriever(handler, **kwargs):
    transport = httpx.MockTransport(handler)
    return TavilyWebRetriever(api_key="tvly-test", transport=transport, **kwargs)


def test_request_shape_and_response_parsing():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/a",
                        "title": "Doc A",
                        "content": "Paris is the capital of France.",
                        "score": 0.9,
                        "published_date": "2025-04-01",
                    }
                ]
            },
        )

    retriever = _make_retriever(handler)
    results = retriever.retrieve("capital of France", top_k=3)

    assert len(results) == 1
    assert results[0].document.uri == "https://example.com/a"
    assert results[0].document.effective_date == "2025-04-01"
    assert "Paris" in results[0].snippet
    body = captured["body"]
    assert body["query"] == "capital of France"
    assert body["max_results"] == 3
    assert body["api_key"] == "tvly-test"


def test_returns_empty_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "server"})

    assert _make_retriever(handler).retrieve("x") == []


def test_returns_empty_when_no_api_key():
    transport = httpx.MockTransport(lambda r: httpx.Response(200))
    retriever = TavilyWebRetriever(api_key="", transport=transport)
    assert retriever.retrieve("anything") == []


def test_returns_empty_for_empty_query():
    retriever = _make_retriever(lambda r: httpx.Response(200, json={"results": []}))
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
