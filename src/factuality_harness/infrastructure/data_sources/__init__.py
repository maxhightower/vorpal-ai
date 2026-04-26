"""Data-source connectors. Each connector adapts a backend (warehouse,
REST API, file collection) into a uniform shape: introspect into a
``DataCatalogEntry``, expose its native handle for tools to query."""

from .base import DataSource
from .warehouse_duckdb import DuckDBWarehouseSource
from .sec_edgar import SECEdgarSource

__all__ = ["DataSource", "DuckDBWarehouseSource", "SECEdgarSource"]
