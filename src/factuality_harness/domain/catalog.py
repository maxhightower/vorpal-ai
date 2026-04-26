"""Data catalog: typed metadata for sources the harness is allowed to query.

A ``DataCatalogEntry`` describes ONE data source — a warehouse table set,
a public API, a file collection. The discovery layer (``DataSourceRouter``)
selects entries from the catalog based on a question + claim type, and the
LLM-assisted translator uses the entry's schema to write payloads against
the source.

What lives here vs. elsewhere:

 - **Here (domain)**: the typed shape of an entry. No I/O, no SDK code.
 - **infrastructure/data_sources/**: per-source connectors that introspect
   live sources and return a populated entry.
 - **application/data_source_router**: discovery + selection logic.

Source kinds are deliberately enumerated rather than free-form so the
router and translator can reason about them statically.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceKind(str, Enum):
    WAREHOUSE = "warehouse"      # SQL-queryable table set
    DOCUMENT_STORE = "document_store"  # text retriever
    REST_API = "rest_api"        # remote JSON/XML API
    FILE_COLLECTION = "file_collection"  # local CSV/JSON/Parquet
    KNOWLEDGE_GRAPH = "knowledge_graph"


class AccessPolicy(str, Enum):
    PUBLIC = "public"            # no auth, no PII
    AUTHENTICATED = "authenticated"  # auth required, no PII
    PII_RESTRICTED = "pii_restricted"  # PII present; needs explicit consent


class TableSchema(BaseModel):
    """Lightweight schema description for one table."""

    name: str
    description: str | None = None
    columns: dict[str, str] = Field(default_factory=dict)  # name -> dtype
    row_count: int | None = None
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)


class DataCatalogEntry(BaseModel):
    """Declarative metadata for one source. Connectors return one of these."""

    source_id: str
    kind: SourceKind
    description: str
    access_policy: AccessPolicy = AccessPolicy.PUBLIC
    freshness_window_days: int | None = None  # None = not time-sensitive
    tables: list[TableSchema] = Field(default_factory=list)
    rest_endpoints: list[str] = Field(default_factory=list)
    document_count: int | None = None
    sample_query: str | None = None
    keywords: list[str] = Field(default_factory=list)
    introspected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def column_names(self) -> list[str]:
        out: list[str] = []
        for t in self.tables:
            out.extend(t.columns.keys())
        return out


class DataCatalog(BaseModel):
    """In-memory registry of catalog entries. Connectors register here at
    startup; the router queries it per request."""

    entries: dict[str, DataCatalogEntry] = Field(default_factory=dict)

    def register(self, entry: DataCatalogEntry) -> None:
        self.entries[entry.source_id] = entry

    def get(self, source_id: str) -> DataCatalogEntry | None:
        return self.entries.get(source_id)

    def all(self) -> list[DataCatalogEntry]:
        return list(self.entries.values())

    def by_kind(self, kind: SourceKind) -> list[DataCatalogEntry]:
        return [e for e in self.entries.values() if e.kind == kind]
