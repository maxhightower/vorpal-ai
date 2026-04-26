"""DuckDB warehouse connector.

Owns a DuckDB connection (in-memory by default), introspects its tables /
columns / row counts / sample rows into a ``DataCatalogEntry``, and
exposes the connection so ``DuckDBSqlExecutor`` can run queries against
it.

This is the reference implementation for Phase A of the data-acquisition
roadmap: one real connector behind the ``DataSource`` protocol. Other
warehouse families (Postgres, Snowflake, BigQuery) follow the same shape.
"""

from __future__ import annotations

import re
from typing import Any

import duckdb

from ...domain.catalog import (
    AccessPolicy,
    DataCatalogEntry,
    SourceKind,
    TableSchema,
)


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise ValueError(
            f"Invalid identifier {name!r}; allowed: letters, digits, underscore."
        )
    return name


def _python_to_sql_type(value: Any) -> str:
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "BIGINT"
    if isinstance(value, float):
        return "DOUBLE"
    return "VARCHAR"


class DuckDBWarehouseSource:
    def __init__(
        self,
        *,
        source_id: str,
        description: str,
        connection: duckdb.DuckDBPyConnection | None = None,
        access_policy: AccessPolicy = AccessPolicy.AUTHENTICATED,
        freshness_window_days: int | None = None,
        keywords: list[str] | None = None,
        sample_row_limit: int = 3,
    ) -> None:
        self.source_id = _validate_identifier(source_id)
        self._description = description
        self._conn = connection or duckdb.connect(":memory:")
        self._access_policy = access_policy
        self._freshness_window_days = freshness_window_days
        self._keywords = list(keywords or [])
        self._sample_row_limit = sample_row_limit

    # ------------------------------------------------------------------
    # Loading helpers (used by tests / fixtures to populate fresh warehouses)
    # ------------------------------------------------------------------

    def load_table(self, name: str, rows: list[dict[str, Any]]) -> None:
        """Create a table from a list-of-dicts. Identifiers are validated;
        values flow through parameter binding."""
        table = _validate_identifier(name)
        if not rows:
            self._conn.execute(
                f'CREATE OR REPLACE TABLE "{table}" (placeholder INTEGER)'
            )
            return

        cols = list(rows[0].keys())
        for c in cols:
            _validate_identifier(c)

        col_types: dict[str, str] = {}
        for c in cols:
            sample = next((r.get(c) for r in rows if r.get(c) is not None), None)
            col_types[c] = _python_to_sql_type(sample)

        column_defs = ", ".join(f'"{c}" {col_types[c]}' for c in cols)
        self._conn.execute(f'CREATE OR REPLACE TABLE "{table}" ({column_defs})')

        placeholders = ", ".join(["?"] * len(cols))
        insert_sql = (
            f'INSERT INTO "{table}" ('
            + ", ".join(f'"{c}"' for c in cols)
            + f") VALUES ({placeholders})"
        )
        self._conn.executemany(
            insert_sql, [tuple(r.get(c) for c in cols) for r in rows]
        )

    # ------------------------------------------------------------------
    # DataSource protocol
    # ------------------------------------------------------------------

    def handle(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    def introspect(self) -> DataCatalogEntry:
        """Walk the connection's catalog and produce a ``DataCatalogEntry``.

        Uses DuckDB's ``information_schema.tables`` / ``columns`` so this
        works against any DuckDB connection — including ones populated by
        attaching external Postgres/Parquet sources.
        """
        tables: list[TableSchema] = []
        rows = self._conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        for (table_name,) in rows:
            columns_rows = self._conn.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'main' AND table_name = ? "
                "ORDER BY ordinal_position",
                [table_name],
            ).fetchall()
            columns = {col_name: dtype for col_name, dtype in columns_rows}

            try:
                row_count = int(
                    self._conn.execute(
                        f'SELECT COUNT(*) FROM "{table_name}"'
                    ).fetchone()[0]
                )
            except duckdb.Error:
                row_count = None

            sample_rows: list[dict[str, Any]] = []
            try:
                sample = self._conn.execute(
                    f'SELECT * FROM "{table_name}" LIMIT {self._sample_row_limit}'
                )
                col_names = [d[0] for d in sample.description] if sample.description else []
                sample_rows = [dict(zip(col_names, row)) for row in sample.fetchall()]
            except duckdb.Error:
                pass

            tables.append(
                TableSchema(
                    name=table_name,
                    columns=columns,
                    row_count=row_count,
                    sample_rows=sample_rows,
                )
            )

        sample_query: str | None = None
        if tables:
            t = tables[0]
            sample_query = f'SELECT * FROM "{t.name}" LIMIT 5'

        return DataCatalogEntry(
            source_id=self.source_id,
            kind=SourceKind.WAREHOUSE,
            description=self._description,
            access_policy=self._access_policy,
            freshness_window_days=self._freshness_window_days,
            tables=tables,
            sample_query=sample_query,
            keywords=list(self._keywords),
        )
