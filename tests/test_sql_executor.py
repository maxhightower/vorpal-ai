from __future__ import annotations

import duckdb
import pytest

from factuality_harness.domain.claims import Claim
from factuality_harness.domain.epistemic_types import EpistemicType
from factuality_harness.domain.evidence import SourceQuality, SourceType, SupportStatus
from factuality_harness.infrastructure.tools.base import ToolRequest
from factuality_harness.infrastructure.tools.sql_executor import DuckDBSqlExecutor


def _claim() -> Claim:
    return Claim(
        text="What was Q1 revenue?",
        parent_question="x",
        epistemic_type=EpistemicType.NUMERICAL,
    )


def test_runs_sql_against_in_memory_table():
    tool = DuckDBSqlExecutor()
    req = ToolRequest(
        claim=_claim(),
        context={
            "tables": {
                "sales": [
                    {"quarter": "Q1", "revenue": 100.0},
                    {"quarter": "Q2", "revenue": 150.0},
                ]
            },
            "sql": "SELECT SUM(revenue) AS total FROM sales WHERE quarter = 'Q1'",
        },
    )
    result = tool.run(req)
    assert result.succeeded
    ev = result.evidence[0]
    assert ev.source_type == SourceType.STRUCTURED_DATA
    assert ev.source_quality == SourceQuality.AUTHORITATIVE
    assert ev.supports_claim == SupportStatus.SUPPORTS
    assert ev.normalized_result["row_count"] == 1
    assert ev.normalized_result["rows"][0]["total"] == 100.0


def test_missing_sql_fails_cleanly():
    tool = DuckDBSqlExecutor()
    result = tool.run(ToolRequest(claim=_claim()))
    assert not result.succeeded
    assert "context['sql']" in (result.error or "")


def test_invalid_sql_returns_error():
    tool = DuckDBSqlExecutor()
    req = ToolRequest(
        claim=_claim(),
        context={"sql": "SELECT * FROM does_not_exist"},
    )
    result = tool.run(req)
    assert not result.succeeded
    assert "SQL error" in (result.error or "")


def test_rejects_invalid_table_name():
    tool = DuckDBSqlExecutor()
    with pytest.raises(ValueError):
        tool.register_table("bad name; DROP TABLE x", [{"a": 1}])


def test_external_connection_is_reused():
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE seeded (x INTEGER)")
    conn.execute("INSERT INTO seeded VALUES (42)")
    tool = DuckDBSqlExecutor(connection=conn)
    result = tool.run(
        ToolRequest(claim=_claim(), context={"sql": "SELECT x FROM seeded"})
    )
    assert result.succeeded
    assert result.evidence[0].normalized_result["rows"][0]["x"] == 42
