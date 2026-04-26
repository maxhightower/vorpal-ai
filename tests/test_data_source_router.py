"""DataSourceRouter tests: keyword, LLM-backed, and pipeline integration."""

from __future__ import annotations

import json

from factuality_harness.application.data_source_router import (
    KeywordDataSourceRouter,
    LLMDataSourceRouter,
    NullDataSourceRouter,
)
from factuality_harness.application.pipeline import (
    FactualityPipeline,
    PipelineRequest,
)
from factuality_harness.domain.catalog import (
    DataCatalog,
    DataCatalogEntry,
    SourceKind,
    TableSchema,
)
from factuality_harness.infrastructure.llm.base import LLMRequest
from factuality_harness.infrastructure.llm.mock_llm import MockLLM
from factuality_harness.infrastructure.storage.repository import (
    InMemoryAuditRepository,
)


def _finance_catalog() -> DataCatalog:
    catalog = DataCatalog()
    catalog.register(
        DataCatalogEntry(
            source_id="financials",
            kind=SourceKind.WAREHOUSE,
            description="Quarterly revenue and gross margin per company.",
            tables=[
                TableSchema(
                    name="quarterly_revenue",
                    columns={
                        "company": "VARCHAR",
                        "quarter": "VARCHAR",
                        "revenue": "DOUBLE",
                    },
                )
            ],
            keywords=["revenue", "quarterly", "earnings"],
        )
    )
    catalog.register(
        DataCatalogEntry(
            source_id="weather",
            kind=SourceKind.REST_API,
            description="Daily temperature and precipitation by station.",
            keywords=["weather", "temperature", "precipitation"],
        )
    )
    return catalog


# ---------------------------------------------------------------------------
# Null router
# ---------------------------------------------------------------------------


def test_null_router_returns_nothing():
    out = NullDataSourceRouter().route(
        question="anything", claims=[], catalog=_finance_catalog()
    )
    assert out == []


# ---------------------------------------------------------------------------
# Keyword router
# ---------------------------------------------------------------------------


def test_keyword_router_picks_finance_for_revenue_question():
    catalog = _finance_catalog()
    out = KeywordDataSourceRouter().route(
        question="What was Q1 revenue last quarter?",
        claims=[],
        catalog=catalog,
    )
    assert [e.source_id for e in out] == ["financials"]


def test_keyword_router_returns_empty_when_no_match():
    catalog = _finance_catalog()
    out = KeywordDataSourceRouter().route(
        question="Tell me a joke about giraffes.",
        claims=[],
        catalog=catalog,
    )
    assert out == []


def test_keyword_router_respects_max_sources():
    catalog = _finance_catalog()
    catalog.register(
        DataCatalogEntry(
            source_id="financials_v2",
            kind=SourceKind.WAREHOUSE,
            description="Audited quarterly revenue restated.",
            keywords=["revenue", "quarterly"],
        )
    )
    out = KeywordDataSourceRouter().route(
        question="What was quarterly revenue?",
        claims=[],
        catalog=catalog,
        max_sources=1,
    )
    assert len(out) == 1


# ---------------------------------------------------------------------------
# LLM-backed router
# ---------------------------------------------------------------------------


def test_llm_router_uses_well_formed_response():
    payload = json.dumps({"selected": ["financials"]})
    llm = MockLLM(responder=lambda req: payload)
    out = LLMDataSourceRouter(llm=llm).route(
        question="What was revenue?",
        claims=[],
        catalog=_finance_catalog(),
    )
    assert [e.source_id for e in out] == ["financials"]


def test_llm_router_drops_unknown_source_ids():
    payload = json.dumps({"selected": ["financials", "made_up_source"]})
    llm = MockLLM(responder=lambda req: payload)
    out = LLMDataSourceRouter(llm=llm).route(
        question="x", claims=[], catalog=_finance_catalog()
    )
    assert [e.source_id for e in out] == ["financials"]


def test_llm_router_falls_back_on_garbage():
    """Garbled output -> use the fallback (default = keyword router)."""
    llm = MockLLM(responder=lambda req: "nope, not json at all")
    out = LLMDataSourceRouter(llm=llm).route(
        question="What was quarterly revenue?",
        claims=[],
        catalog=_finance_catalog(),
    )
    assert [e.source_id for e in out] == ["financials"]


def test_llm_router_falls_back_on_exception():
    class _Boom:
        name = "boom"

        def complete(self, request: LLMRequest):
            raise RuntimeError("nope")

    out = LLMDataSourceRouter(llm=_Boom()).route(
        question="What was revenue?",
        claims=[],
        catalog=_finance_catalog(),
    )
    # Fallback is the keyword router; revenue question matches financials.
    assert [e.source_id for e in out] == ["financials"]


def test_llm_router_returns_empty_for_empty_catalog():
    llm = MockLLM(responder=lambda req: '{"selected": ["x"]}')
    out = LLMDataSourceRouter(llm=llm).route(
        question="x", claims=[], catalog=DataCatalog()
    )
    assert out == []


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_pipeline_with_no_catalog_records_no_consulted_sources():
    pipeline = FactualityPipeline(audit_repo=InMemoryAuditRepository())
    final = pipeline.run(
        PipelineRequest(question="What is the percentage increase from 100 to 125?")
    )
    trace = pipeline.audit_repo.get(final.audit_id)
    assert trace.data_sources_consulted == []


def test_pipeline_with_catalog_routes_and_records_consulted_sources():
    catalog = _finance_catalog()
    pipeline = FactualityPipeline(
        audit_repo=InMemoryAuditRepository(),
        data_catalog=catalog,
        data_source_router=KeywordDataSourceRouter(),
    )
    final = pipeline.run(
        PipelineRequest(
            question="What is the percentage increase from 100 to 125 in revenue?"
        )
    )
    trace = pipeline.audit_repo.get(final.audit_id)
    assert [e.source_id for e in trace.data_sources_consulted] == ["financials"]


def test_pipeline_data_sources_appear_in_ctx_for_translator():
    """The discovered source should be summarized into ``ctx['data']`` so a
    downstream LLM-assisted translator can see it. We verify by checking
    the audit trace + data field shape — the translator itself is a
    separate test surface."""
    catalog = _finance_catalog()
    pipeline = FactualityPipeline(
        audit_repo=InMemoryAuditRepository(),
        data_catalog=catalog,
        data_source_router=KeywordDataSourceRouter(),
    )
    final = pipeline.run(
        PipelineRequest(question="What is quarterly revenue?")
    )
    trace = pipeline.audit_repo.get(final.audit_id)
    assert trace.data_sources_consulted, (
        "expected at least one consulted source in the audit trace"
    )
