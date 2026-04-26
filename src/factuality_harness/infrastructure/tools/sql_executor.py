"""DuckDB-backed SQL executor.

Evidence comes from explicit SQL — a deterministic query you (or a domain
module) supply. This tool does NOT translate natural-language claims to SQL;
that's a separate concern (an LLM-backed translator could write to
``request.context["sql"]`` upstream). Keeping translation out of this tool
preserves the harness's "verification is deterministic" property.

Expected context shape::

    request.context["sql"] = "SELECT ..."
    # Optional — register additional in-memory tables for the query:
    request.context["tables"] = {
        "<table_name>": [{"col": "v", ...}, ...]   # rows as list-of-dicts
    }

Each query result row is captured in ``Evidence.normalized_result`` so
downstream verdict logic can read column values directly.
"""

from __future__ import annotations

import re
from typing import Any

import duckdb

from ...domain.evidence import (
    FreshnessStatus,
    SourceQuality,
    SourceType,
    SupportStatus,
)
from .base import Tool, ToolRequest, ToolResult, build_evidence


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: str) -> str:
    """Allow only conservative table/column names. Prevents SQL injection
    via dynamic identifiers (DuckDB's parameter binding does not cover
    identifiers, only values)."""
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


class DuckDBSqlExecutor:
    name = "sql_executor"

    def __init__(
        self,
        *,
        connection: duckdb.DuckDBPyConnection | None = None,
        max_rows: int = 1000,
    ) -> None:
        self._conn = connection or duckdb.connect(":memory:")
        self._max_rows = max_rows

    def register_table(self, name: str, rows: list[dict[str, Any]]) -> None:
        """Register a table on the executor's default connection. Use
        ``_register_table_on`` to target a different connection (e.g. a
        warehouse handed in via context)."""
        self._register_table_on(self._conn, name, rows)

    @staticmethod
    def _register_table_on(
        conn: duckdb.DuckDBPyConnection,
        name: str,
        rows: list[dict[str, Any]],
    ) -> None:
        table = _validate_identifier(name)

        if not rows:
            conn.execute(
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
        conn.execute(f'CREATE OR REPLACE TABLE "{table}" ({column_defs})')

        placeholders = ", ".join(["?"] * len(cols))
        insert_sql = (
            f'INSERT INTO "{table}" ('
            + ", ".join(f'"{c}"' for c in cols)
            + f") VALUES ({placeholders})"
        )
        conn.executemany(insert_sql, [tuple(r.get(c) for c in cols) for r in rows])

    def run(self, request: ToolRequest) -> ToolResult:
        sql = request.context.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            return ToolResult(
                succeeded=False,
                error="DuckDBSqlExecutor requires request.context['sql'].",
            )

        # If the data-source router selected a warehouse for this request,
        # the pipeline drops its connection here. Use that connection
        # instead of the executor's default in-memory one so the SQL runs
        # against the real warehouse data.
        sql_connection = request.context.get("sql_connection")
        active_conn = sql_connection if sql_connection is not None else self._conn

        tables = request.context.get("tables") or {}
        if isinstance(tables, dict):
            for table_name, rows in tables.items():
                if isinstance(rows, list):
                    # Register against the active connection — registering
                    # against self._conn when a warehouse was passed would
                    # hide tables from the actual query target.
                    self._register_table_on(active_conn, table_name, rows)

        try:
            cursor = active_conn.execute(sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            data = cursor.fetchmany(self._max_rows)
        except (duckdb.Error, ValueError) as e:
            return ToolResult(succeeded=False, error=f"SQL error: {e}")

        rows = [dict(zip(columns, row)) for row in data]

        preview = "; ".join(
            ", ".join(f"{k}={v!r}" for k, v in r.items()) for r in rows[:5]
        )
        summary = f"{len(rows)} row(s) returned"

        return ToolResult(
            evidence=[
                build_evidence(
                    claim_id=request.claim.id,
                    source_type=SourceType.STRUCTURED_DATA,
                    source_name="duckdb",
                    quote_or_result=(
                        f"SQL: {sql.strip()} -> {summary}"
                        + (f". Preview: {preview}" if preview else "")
                    ),
                    normalized_result={
                        "sql": sql.strip(),
                        "columns": columns,
                        "row_count": len(rows),
                        "rows": rows,
                    },
                    supports_claim=(
                        SupportStatus.SUPPORTS if rows else SupportStatus.INSUFFICIENT
                    ),
                    source_quality=SourceQuality.AUTHORITATIVE,
                    freshness=FreshnessStatus.NOT_TIME_SENSITIVE,
                    notes="Deterministic SQL query result.",
                )
            ]
        )


_: Tool = DuckDBSqlExecutor()
