"""Stub SQL executor tool.

A real implementation would route to a structured datastore (DuckDB, Postgres,
warehouse). The MVP keeps the interface so domain modules can declare structured
queries even before a backing store is wired up.
"""

from __future__ import annotations

from .base import Tool, ToolRequest, ToolResult


class SqlExecutorStub:
    name = "sql_executor"

    def run(self, request: ToolRequest) -> ToolResult:
        return ToolResult(
            succeeded=False,
            error="SqlExecutorStub is not implemented. Wire a DuckDB/Postgres backend.",
        )


_: Tool = SqlExecutorStub()
