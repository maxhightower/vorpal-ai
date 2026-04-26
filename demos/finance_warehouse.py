"""End-to-end demo of the finance vertical.

Wires every piece of the data-acquisition stack together for a single
realistic question — *"What was ACME's 2026-Q1 quarterly revenue?"* —
and prints the audit trace so the discovery + translation + execution
chain is visible.

Run::

    .venv/bin/python demos/finance_warehouse.py
"""

from __future__ import annotations

import json
import warnings

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


warnings.filterwarnings("ignore")


# Fixture: synthetic but fully specified — a tiny quarterly revenue table.
ROWS = [
    {"company": "ACME", "quarter": "2026-Q1", "revenue": 250.0},
    {"company": "ACME", "quarter": "2026-Q2", "revenue": 300.0},
    {"company": "ACME", "quarter": "2026-Q3", "revenue": 320.0},
    {"company": "BETA", "quarter": "2026-Q1", "revenue": 120.0},
    {"company": "BETA", "quarter": "2026-Q2", "revenue": 135.0},
]


def _translator_responder(req):
    """Mock LLM stand-in for AnthropicAdapter. Emits a SQL payload for the
    sql_executor whenever the translator asks for one."""
    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
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


def main() -> None:
    # 1. Build a real warehouse and load it.
    warehouse = DuckDBWarehouseSource(
        source_id="financials",
        description="Quarterly revenue per company.",
        keywords=["revenue", "quarterly", "company", "earnings"],
    )
    warehouse.load_table("quarterly_revenue", ROWS)

    # 2. Register it in the catalog (with introspected schema).
    catalog = DataCatalog()
    catalog.register(warehouse.introspect())

    # 3. Wire the pipeline: ACTIVE FinanceModule, keyword discovery,
    #    LLM-assisted translator (mocked), and the warehouse handle.
    pipeline = FactualityPipeline(
        audit_repo=InMemoryAuditRepository(),
        module_registry=ModuleRegistry(modules=[FinanceModule(active=True)]),
        data_catalog=catalog,
        data_source_router=KeywordDataSourceRouter(),
        data_sources={"financials": warehouse},
        tool_input_translator=LLMToolInputTranslator(
            llm=MockLLM(responder=_translator_responder)
        ),
    )

    # 4. Ask a finance question.
    question = "What was ACME's 2026-Q1 quarterly revenue?"
    final = pipeline.run(PipelineRequest(question=question))
    trace = pipeline.audit_repo.get(final.audit_id)

    print(f"Question: {question}\n")

    print("Discovery layer consulted:")
    for entry in trace.data_sources_consulted:
        print(f"  - {entry.source_id} ({entry.kind.value}): {entry.description}")
        for table in entry.tables:
            print(f"      table {table.name!r}: {list(table.columns.keys())}")
    print()

    print("Tools called:")
    for tc in trace.tools_called:
        status = "OK" if tc.succeeded else f"ERR ({(tc.error or '')[:60]})"
        print(f"  - {tc.tool_name}: {status}")
    print()

    print("Evidence:")
    for e in trace.retrieved_evidence:
        print(f"  - [{e.source_type.value}] {e.source_name}")
        print(f"      supports_claim={e.supports_claim.value}")
        if e.normalized_result and "rows" in e.normalized_result:
            print(f"      rows={e.normalized_result['rows']}")
    print()

    print("Verdicts:")
    for v in trace.verdicts:
        print(f"  - {v.verdict.value} (confidence={v.confidence.value})")
        print(f"      rationale: {v.rationale}")
    print()

    print("Final answer (truncated):")
    print(final.answer[:600])
    print()
    print(f"audit_id: {final.audit_id}")


if __name__ == "__main__":
    main()
