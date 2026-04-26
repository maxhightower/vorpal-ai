"""Catalog model + DuckDB warehouse + SEC EDGAR connector tests."""

from __future__ import annotations

import json

import duckdb
import httpx
import pytest

from factuality_harness.domain.catalog import (
    AccessPolicy,
    DataCatalog,
    DataCatalogEntry,
    SourceKind,
    TableSchema,
)
from factuality_harness.infrastructure.data_sources.sec_edgar import (
    SECEdgarSource,
)
from factuality_harness.infrastructure.data_sources.warehouse_duckdb import (
    DuckDBWarehouseSource,
)


# ---------------------------------------------------------------------------
# DataCatalog
# ---------------------------------------------------------------------------


def test_catalog_register_and_lookup():
    catalog = DataCatalog()
    entry = DataCatalogEntry(
        source_id="warehouse_x",
        kind=SourceKind.WAREHOUSE,
        description="x",
    )
    catalog.register(entry)
    assert catalog.get("warehouse_x") is entry
    assert catalog.get("nonexistent") is None
    assert catalog.all() == [entry]


def test_catalog_filter_by_kind():
    catalog = DataCatalog()
    catalog.register(
        DataCatalogEntry(source_id="a", kind=SourceKind.WAREHOUSE, description="x")
    )
    catalog.register(
        DataCatalogEntry(source_id="b", kind=SourceKind.REST_API, description="y")
    )
    assert [e.source_id for e in catalog.by_kind(SourceKind.WAREHOUSE)] == ["a"]
    assert [e.source_id for e in catalog.by_kind(SourceKind.REST_API)] == ["b"]


def test_catalog_entry_column_names_aggregates_across_tables():
    entry = DataCatalogEntry(
        source_id="x",
        kind=SourceKind.WAREHOUSE,
        description="x",
        tables=[
            TableSchema(name="t1", columns={"a": "BIGINT", "b": "VARCHAR"}),
            TableSchema(name="t2", columns={"c": "DOUBLE"}),
        ],
    )
    assert sorted(entry.column_names()) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# DuckDBWarehouseSource
# ---------------------------------------------------------------------------


def test_duckdb_warehouse_introspects_loaded_tables():
    src = DuckDBWarehouseSource(
        source_id="financials",
        description="Test fixture",
        access_policy=AccessPolicy.AUTHENTICATED,
        keywords=["revenue", "quarterly"],
    )
    src.load_table(
        "quarterly_revenue",
        [
            {"company": "ACME", "quarter": "2026-Q1", "revenue": 250.0},
            {"company": "ACME", "quarter": "2026-Q2", "revenue": 300.0},
            {"company": "BETA", "quarter": "2026-Q1", "revenue": 120.0},
        ],
    )
    src.load_table("customers", [{"id": 1, "name": "First"}])

    entry = src.introspect()
    assert entry.source_id == "financials"
    assert entry.kind == SourceKind.WAREHOUSE
    assert entry.access_policy == AccessPolicy.AUTHENTICATED
    assert entry.keywords == ["revenue", "quarterly"]

    table_names = {t.name for t in entry.tables}
    assert table_names == {"quarterly_revenue", "customers"}

    qr = next(t for t in entry.tables if t.name == "quarterly_revenue")
    assert set(qr.columns.keys()) == {"company", "quarter", "revenue"}
    assert qr.row_count == 3
    assert len(qr.sample_rows) <= 3
    assert qr.sample_rows[0]["company"] == "ACME"


def test_duckdb_warehouse_handle_runs_real_queries():
    src = DuckDBWarehouseSource(
        source_id="w",
        description="x",
    )
    src.load_table("nums", [{"x": 1}, {"x": 2}, {"x": 3}])
    handle = src.handle()
    total = handle.execute("SELECT SUM(x) FROM nums").fetchone()[0]
    assert total == 6


def test_duckdb_warehouse_rejects_invalid_identifier():
    src = DuckDBWarehouseSource(source_id="w", description="x")
    with pytest.raises(ValueError):
        src.load_table("bad name; DROP TABLE", [{"a": 1}])
    with pytest.raises(ValueError):
        DuckDBWarehouseSource(source_id="bad-id", description="x")


def test_duckdb_warehouse_external_connection_is_reused():
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE seeded (x INTEGER)")
    conn.execute("INSERT INTO seeded VALUES (42)")
    src = DuckDBWarehouseSource(
        source_id="w", description="x", connection=conn
    )
    entry = src.introspect()
    assert any(t.name == "seeded" for t in entry.tables)


def test_duckdb_warehouse_empty_introspection():
    src = DuckDBWarehouseSource(source_id="w", description="empty")
    entry = src.introspect()
    assert entry.tables == []
    assert entry.sample_query is None


# ---------------------------------------------------------------------------
# SECEdgarSource
# ---------------------------------------------------------------------------


def test_sec_edgar_introspect_is_static_no_network():
    src = SECEdgarSource(
        transport=httpx.MockTransport(
            lambda req: pytest.fail("introspect must not hit the network")
        )
    )
    entry = src.introspect()
    assert entry.kind == SourceKind.REST_API
    assert entry.access_policy == AccessPolicy.PUBLIC
    assert any("sec.gov" in url for url in entry.rest_endpoints)
    # Useful for the discovery layer.
    assert "revenue" in entry.keywords
    assert "cik" in entry.keywords


def test_sec_edgar_handle_calls_company_facts_endpoint():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "cik": 320193,
                "entityName": "Apple Inc.",
                "facts": {"us-gaap": {}},
            },
        )

    src = SECEdgarSource(transport=httpx.MockTransport(handler))
    facts = src.handle().get_company_facts(320193)
    assert facts["entityName"] == "Apple Inc."
    assert "CIK0000320193.json" in captured["url"]
    # SEC requires a User-Agent header.
    assert captured["headers"].get("user-agent")


def test_sec_edgar_handle_propagates_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    src = SECEdgarSource(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        src.handle().get_company_facts(999999999)
