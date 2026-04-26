"""In-memory keyword retriever over documents supplied at request time.

Deliberately simple: token overlap with optional sentence-level snippet
extraction. The interface is identical to a real vector retriever, so the
pipeline doesn't change when one is swapped in.
"""

from __future__ import annotations

import re

from .base import Document, RetrievalResult, Retriever

_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z\-']+")
_SENT = re.compile(r"(?<=[\.!?])\s+")

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "by", "with", "from", "as",
    "and", "or", "but", "if", "then", "this", "that", "these", "those",
    "it", "its", "do", "does", "did", "have", "has", "had", "not", "no",
    "can", "could", "will", "would", "should", "may", "might", "must",
}


def _tokens(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(s) if t.lower() not in _STOPWORDS and len(t) > 2]


class LocalDocumentRetriever:
    def __init__(self, documents: list[Document] | None = None) -> None:
        self._docs: list[Document] = list(documents or [])

    def add(self, doc: Document) -> None:
        self._docs.append(doc)

    def replace(self, documents: list[Document]) -> None:
        self._docs = list(documents)

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        q_tokens = set(_tokens(query))
        if not q_tokens:
            return []

        results: list[RetrievalResult] = []
        for doc in self._docs:
            best_snippet = ""
            best_score = 0.0
            best_offset = 0
            offset = 0
            for sent in _SENT.split(doc.text):
                s_tokens = set(_tokens(sent))
                if not s_tokens:
                    offset += len(sent) + 1
                    continue
                overlap = len(q_tokens & s_tokens)
                if overlap == 0:
                    offset += len(sent) + 1
                    continue
                # Normalize by sentence size to prefer focused matches.
                score = overlap / (1.0 + 0.1 * len(s_tokens))
                if score > best_score:
                    best_score = score
                    best_snippet = sent.strip()
                    best_offset = offset
                offset += len(sent) + 1

            if best_score > 0:
                results.append(
                    RetrievalResult(
                        document=doc,
                        score=best_score,
                        snippet=best_snippet,
                        snippet_offset=best_offset,
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


# Static type compatibility check.
_: Retriever = LocalDocumentRetriever()
