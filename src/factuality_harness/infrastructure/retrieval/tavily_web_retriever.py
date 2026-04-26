"""Tavily-backed web retriever.

Calls Tavily's ``/search`` endpoint via raw httpx so it can be unit-tested
with ``httpx.MockTransport``. Tavily was chosen because the response shape
maps cleanly onto the harness's ``RetrievalResult`` (URL, title, content,
optional published date), but any provider with a similar shape can be
swapped in by subclassing or templating the response parser.

Auth: pass ``api_key`` or set ``TAVILY_API_KEY`` in the environment.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import Document, RetrievalResult, Retriever


_TAVILY_URL = "https://api.tavily.com/search"


class TavilyWebRetriever:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = _TAVILY_URL,
        timeout: float = 15.0,
        transport: httpx.BaseTransport | None = None,
        max_results: int = 5,
    ) -> None:
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        self._url = base_url
        self._client = httpx.Client(timeout=timeout, transport=transport)
        self._max_results = max_results

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        if not self._api_key:
            return []
        if not query.strip():
            return []

        body: dict[str, Any] = {
            "api_key": self._api_key,
            "query": query,
            "max_results": min(top_k, self._max_results),
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
        }
        try:
            response = self._client.post(self._url, json=body)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError):
            return []

        raw_results = data.get("results") or []
        out: list[RetrievalResult] = []
        for item in raw_results[:top_k]:
            url = item.get("url") or ""
            title = item.get("title") or url or "<untitled>"
            content = item.get("content") or ""
            published_date = item.get("published_date")
            score = float(item.get("score") or 0.0)
            out.append(
                RetrievalResult(
                    document=Document(
                        name=title,
                        text=content,
                        uri=url or None,
                        effective_date=published_date,
                    ),
                    score=score,
                    snippet=(content[:400] if content else title),
                    snippet_offset=0,
                )
            )
        return out


_: Retriever = TavilyWebRetriever.__new__(TavilyWebRetriever)
