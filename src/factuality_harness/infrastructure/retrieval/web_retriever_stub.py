"""Web retriever stub.

Returns no results. The interface exists so a real search backend can be wired
in later (e.g. Bing, Tavily, an internal search service).
"""

from __future__ import annotations

from .base import RetrievalResult, Retriever


class WebRetrieverStub:
    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        return []


_: Retriever = WebRetrieverStub()
