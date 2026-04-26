"""Common protocol for data-source connectors.

A ``DataSource`` exposes two things:

 - ``introspect()`` returns a fully-populated ``DataCatalogEntry`` so the
   discovery layer + LLM translator can reason about what the source
   contains without touching it.
 - ``handle()`` returns the source's native client (a DuckDB connection,
   an httpx client, a file path) for the actual tool call. The harness
   keeps this opaque — only the tool that consumes the source needs to
   know its concrete type.
"""

from __future__ import annotations

from typing import Any, Protocol

from ...domain.catalog import DataCatalogEntry


class DataSource(Protocol):
    source_id: str

    def introspect(self) -> DataCatalogEntry: ...

    def handle(self) -> Any: ...
