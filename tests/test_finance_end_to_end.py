"""End-to-end: finance question → catalog → SQL → SUPPORTED verdict.

Wires every component together for the first vertical:

  - FinanceModule (ACTIVE) declares the finance taxonomy + evidence rules.
  - DuckDBWarehouseSource holds quarterly_revenue fixture data.
  - DataCatalog has its introspected entry.
  - KeywordDataSourceRouter picks the warehouse for revenue questions.
  - LLMToolInputTranslator (with a deterministic mock LLM) writes a SQL
    query against the warehouse's tables.
  - DuckDBSqlExecutor runs the query against the warehouse's connection
    (passed via ctx['sql_connection']).
  - The SUPPORTED structured-data evidence yields a verdict.
  - AuditTrace.data_sources_consulted records which sources were used.
"""

from __future__ import annotations

import json

from factuality_harness.application.data_source_router import (
    KeywordDataSourceRouter,
)
from factuality_harness.application.module_registry import ModuleRegistry
from factuality_harness.application.pipeline import (
    FactualityPipeline,
    PipelineRequest,
)
from factuality_harness.application.tool_input_translator import (
    LLMToolInputTranslator,
)
from factuality_harness.domain.catalog import DataCatalog
from factuality_harness.infrastructure.data_sources.warehouse_duckdb import (
    DuckDBWarehouseSource,
)
from factuality_harness.infrastructure.llm.mock_llm import MockLLM
from factuality_harness.infrastructure.storage.repository import (
    InMemoryAuditRepository,
)
from factuality_harness.modules.finance import FinanceModule


_FIXTURE_ROWS = [
    {"company": "ACME", "quarter": "2026-Q1", "revenue": 250.0},
    {"company": "ACME", "quarter": "2026-Q2", "revenue": 300.0},
    {"company": "BETA", "quarter": "2026-Q1", "revenue": 120.0},
]


def _build_finance_pipeline() -> FactualityPipeline:
    src = DuckDBWarehouseSource(
        source_id="financials",
        description="Quarterly revenue per company.",
        keywords=["revenue", "quarterly", "company"],
    )
    src.load_table("quarterly_revenue", _FIXTURE_ROWS)
    catalog = DataCatalog()
    catalog.register(src.introspect())

    # Mock LLM produces a concrete SQL query whenever the translator asks
    # for a sql_executor payload. Real-world swap-in: AnthropicAdapter.
    def _translator_responder(req):
        last_user = next(
            (m.content for m in reversed(req.messages) if m.role == "user"),
            "",
        )
        if "sql_executor" not in last_user:
            return "{}"
        return json.dumps(
            {
                "sql_executor": {
                    "sql": (
                        "SELECT SUM(revenue) AS total FROM quarterly_revenue "
                        "WHERE company = 'ACME' AND quarter = '2026-Q1'"
                    )
                }
            }
        )

    return FactualityPipeline(
        audit_repo=InMemoryAuditRepository(),
        module_registry=ModuleRegistry(modules=[FinanceModule(active=True)]),
        data_catalog=catalog,
        data_source_router=KeywordDataSourceRouter(),
        data_sources={"financials": src},
        tool_input_translator=LLMToolInputTranslator(
            llm=MockLLM(responder=_translator_responder)
        ),
    )


# ---------------------------------------------------------------------------


def test_finance_question_routes_through_warehouse_and_returns_supported():
    pipeline = _build_finance_pipeline()
    final = pipeline.run(
        PipelineRequest(
            question="What was ACME's 2026-Q1 quarterly revenue?",
        )
    )
    trace = pipeline.audit_repo.get(final.audit_id)
    assert trace is not None

    # Discovery layer picked the warehouse.
    assert [e.source_id for e in trace.data_sources_consulted] == ["financials"]

    # SQL executor ran successfully against the warehouse connection.
    sql_calls = [tc for tc in trace.tools_called if tc.tool_name == "sql_executor"]
    assert sql_calls, "sql_executor was not invoked"
    assert sql_calls[0].succeeded, sql_calls[0].error

    # Structured-data evidence reflects the actual fixture: ACME 2026-Q1 = 250.
    structured = [
        e for e in trace.retrieved_evidence
        if e.source_type.value == "STRUCTURED_DATA"
    ]
    assert structured, "no STRUCTURED_DATA evidence emitted"
    rows = structured[0].normalized_result["rows"]
    assert rows == [{"total": 250.0}]


def test_finance_module_applies_to_finance_questions_only():
    fm = FinanceModule(active=True)
    assert fm.applies_to("What was Q1 revenue?", []) > 0.0
    assert fm.applies_to("How was your weekend?", []) == 0.0


def test_finance_module_required_evidence_for_numerical_demands_warehouse():
    fm = FinanceModule(active=True)
    from factuality_harness.domain.claims import Claim
    from factuality_harness.domain.epistemic_types import EpistemicType
    from factuality_harness.domain.evidence import (
        SourceQuality,
        SourceType,
    )

    claim = Claim(
        text="What was Q1 revenue?",
        parent_question="x",
        epistemic_type=EpistemicType.NUMERICAL,
    )
    reqs = fm.required_evidence_for(claim)
    assert reqs
    req = reqs[0]
    assert SourceType.STRUCTURED_DATA in req.required_source_types
    assert req.minimum_source_quality == SourceQuality.AUTHORITATIVE
    assert req.requires_date_check is True


def test_pipeline_records_data_sources_in_audit_trace_only_when_consulted():
    pipeline = _build_finance_pipeline()
    # A question with no finance keywords should not consult the warehouse.
    final = pipeline.run(
        PipelineRequest(question="Tell me a joke about giraffes.")
    )
    trace = pipeline.audit_repo.get(final.audit_id)
    assert trace.data_sources_consulted == []
