from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class Document(BaseModel):
    name: str
    text: str
    uri: str | None = None
    effective_date: str | None = None  # ISO date string; freshness checking uses this


class RetrievalResult(BaseModel):
    document: Document
    score: float
    snippet: str
    snippet_offset: int = 0


class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]: ...
