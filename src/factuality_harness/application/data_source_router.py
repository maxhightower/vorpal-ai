"""Discovery layer: pick which catalog entries are relevant to a request.

Same opt-in pattern as the LLM-backed claim decomposer + tool-input
translator: a deterministic keyword router runs by default, an LLM-backed
router can be plugged in behind the same protocol when configured.

The router does not execute against sources — it produces a list of
catalog entries that downstream stages (the LLM-assisted translator, the
SQL executor, the audit trace) can consume.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from ..domain.catalog import DataCatalog, DataCatalogEntry
from ..infrastructure.llm.base import LLM, LLMMessage, LLMRequest


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "by", "with", "from", "as",
    "and", "or", "but", "if", "then", "this", "that", "these", "those",
    "it", "its", "do", "does", "did", "have", "has", "had", "not", "no",
    "can", "could", "will", "would", "should", "may", "might", "must",
    "what", "which", "who", "whom", "whose", "where", "when", "why",
    "how",
}


def _tokens(s: str) -> set[str]:
    return {
        t.lower()
        for t in _TOKEN_RE.findall(s)
        if len(t) > 2 and t.lower() not in _STOPWORDS
    }


# ---------------------------------------------------------------------------
# Protocol + implementations
# ---------------------------------------------------------------------------


class DataSourceRouter(Protocol):
    def route(
        self,
        *,
        question: str,
        claims: list,
        catalog: DataCatalog,
        max_sources: int = 3,
    ) -> list[DataCatalogEntry]: ...


class KeywordDataSourceRouter:
    """Default deterministic router. Scores each catalog entry by overlap
    of the question/claim tokens against the entry's description, keyword
    list, table names, column names, and sample-row values.

    Returns up to ``max_sources`` entries above a minimum score so a
    request that nothing in the catalog matches comes back empty (instead
    of forcing the translator to hallucinate against an irrelevant
    source)."""

    def __init__(self, *, min_score: float = 0.05) -> None:
        self._min_score = min_score

    def route(
        self,
        *,
        question: str,
        claims: list,
        catalog: DataCatalog,
        max_sources: int = 3,
    ) -> list[DataCatalogEntry]:
        query_tokens = _tokens(question)
        for c in claims:
            query_tokens |= _tokens(getattr(c, "text", "") or "")
        if not query_tokens:
            return []

        scored: list[tuple[float, DataCatalogEntry]] = []
        for entry in catalog.all():
            entry_tokens = self._entry_tokens(entry)
            if not entry_tokens:
                continue
            overlap = len(query_tokens & entry_tokens)
            score = overlap / len(entry_tokens)
            if score >= self._min_score:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:max_sources]]

    @staticmethod
    def _entry_tokens(entry: DataCatalogEntry) -> set[str]:
        bits: list[str] = [entry.description]
        bits.extend(entry.keywords)
        for table in entry.tables:
            bits.append(table.name)
            bits.extend(table.columns.keys())
            for row in table.sample_rows:
                for v in row.values():
                    if isinstance(v, str):
                        bits.append(v)
        return _tokens(" ".join(bits))


class NullDataSourceRouter:
    """Never returns sources. Default when no catalog is wired."""

    def route(
        self,
        *,
        question: str,
        claims: list,
        catalog: DataCatalog,
        max_sources: int = 3,
    ) -> list[DataCatalogEntry]:
        return []


_LLM_SYSTEM_PROMPT = """\
You select which data sources are relevant to a user question.

Rules:
1. Output ONLY a JSON object: {"selected": [<source_id>, ...]}.
2. Use ONLY source_id values that appear in the supplied catalog.
3. Select fewer rather than more — only sources that genuinely apply.
4. Never invent source ids. Never include explanatory prose.
"""


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("{"):
        try:
            out = json.loads(text)
            return out if isinstance(out, dict) else None
        except json.JSONDecodeError:
            pass
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            return None
    return None


class LLMDataSourceRouter:
    """LLM-backed selector with a hard fallback to the keyword router.

    Same shape as ``LLMClaimDecomposer`` and ``LLMToolInputTranslator``:
    strict JSON output, validated against the catalog, fallback on any
    parse error or exception.
    """

    def __init__(
        self,
        *,
        llm: LLM,
        fallback: DataSourceRouter | None = None,
        max_tokens: int = 512,
    ) -> None:
        self._llm = llm
        self._fallback = fallback or KeywordDataSourceRouter()
        self._max_tokens = max_tokens

    def route(
        self,
        *,
        question: str,
        claims: list,
        catalog: DataCatalog,
        max_sources: int = 3,
    ) -> list[DataCatalogEntry]:
        if not catalog.all():
            return []

        catalog_summary = [
            {
                "source_id": e.source_id,
                "kind": e.kind.value,
                "description": e.description,
                "tables": [t.name for t in e.tables],
                "keywords": e.keywords,
            }
            for e in catalog.all()
        ]
        user_payload = json.dumps(
            {"question": question, "catalog": catalog_summary},
            default=str,
            indent=2,
        )

        try:
            response = self._llm.complete(
                LLMRequest(
                    messages=[
                        LLMMessage(role="system", content=_LLM_SYSTEM_PROMPT),
                        LLMMessage(role="user", content=user_payload),
                    ],
                    temperature=0.0,
                    max_tokens=self._max_tokens,
                )
            )
        except Exception:
            return self._fallback.route(
                question=question,
                claims=claims,
                catalog=catalog,
                max_sources=max_sources,
            )

        parsed = _extract_json(response.text)
        if not isinstance(parsed, dict):
            return self._fallback.route(
                question=question,
                claims=claims,
                catalog=catalog,
                max_sources=max_sources,
            )
        ids = parsed.get("selected")
        if not isinstance(ids, list):
            return self._fallback.route(
                question=question,
                claims=claims,
                catalog=catalog,
                max_sources=max_sources,
            )

        out: list[DataCatalogEntry] = []
        for sid in ids[:max_sources]:
            entry = catalog.get(sid) if isinstance(sid, str) else None
            if entry is not None:
                out.append(entry)
        return out
