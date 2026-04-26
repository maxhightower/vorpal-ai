"""End-to-end factuality pipeline.

This is the orchestrator. It is intentionally explicit — every step is a method
call that produces inspectable state. It does not call LLM APIs or external
services directly; those live behind ``infrastructure`` ports passed in via
construction.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..domain.audit import AuditTrace
from ..domain.claims import Claim
from ..domain.evidence import EvidenceTable
from ..domain.verdicts import FinalAnswer
from ..infrastructure.retrieval.base import Document, Retriever
from ..infrastructure.retrieval.local_document_retriever import LocalDocumentRetriever
from ..infrastructure.storage.repository import (
    AuditRepository,
    InMemoryAuditRepository,
)
from ..infrastructure.tools.ab_test import ABTestCausalTool
from ..infrastructure.tools.base import Tool
from ..infrastructure.tools.calculator import CalculatorTool
from ..infrastructure.tools.causal_inference import CausalInferenceTool
from ..infrastructure.tools.causal_model_stub import CausalModelStub
from ..infrastructure.tools.forecast import ForecastTool
from ..infrastructure.tools.optimizer import LinearOptimizerTool
from ..infrastructure.tools.python_executor import LocalSubprocessPythonExecutor
from ..infrastructure.tools.rule_engine import RuleEngineTool
from ..infrastructure.tools.sql_executor import DuckDBSqlExecutor
from ..infrastructure.tools.theorem_prover_stub import TheoremProverStub
from .claim_classifier import ClaimClassifier, RuleBasedClaimClassifier
from .claim_decomposer import ClaimDecomposer, RuleBasedClaimDecomposer
from .contradiction_checker import (
    ContradictionDetector,
    LexicalContradictionDetector,
    check_contradictions,
)
from .draft_generator import DraftGenerator, TemplateDraftGenerator
from .evidence_builder import EvidenceBuilder
from .final_verifier import (
    TemplateDraftClaimExtractor,
    revise_answer,
    verify_draft_claims,
)
from .data_source_router import DataSourceRouter, NullDataSourceRouter
from .module_registry import ModuleRegistry
from .router import ToolRouter
from .tool_input_translator import (
    TOOL_INPUT_SPECS,
    NullToolInputTranslator,
    ToolInputTranslator,
)
from .uncertainty_calibrator import assign_verdicts
from ..domain.catalog import DataCatalog


class PipelineRequest(BaseModel):
    question: str
    domain_hint: str | None = None
    documents: list[Document] = []
    # Free-form context handed verbatim to every tool. Use this for structured
    # payloads (experimental data, time series, reference tables) that do not
    # fit naturally into ``documents``.
    extra_context: dict[str, Any] = Field(default_factory=dict)


class FactualityPipeline:
    def __init__(
        self,
        *,
        decomposer: ClaimDecomposer | None = None,
        classifier: ClaimClassifier | None = None,
        router: ToolRouter | None = None,
        evidence_builder: EvidenceBuilder | None = None,
        draft_generator: DraftGenerator | None = None,
        retriever: Retriever | None = None,
        audit_repo: AuditRepository | None = None,
        module_registry: ModuleRegistry | None = None,
        tools: dict[str, Tool] | None = None,
        tool_input_translator: ToolInputTranslator | None = None,
        contradiction_detector: ContradictionDetector | None = None,
        data_catalog: DataCatalog | None = None,
        data_source_router: DataSourceRouter | None = None,
        data_sources: dict[str, Any] | None = None,
    ) -> None:
        self.decomposer = decomposer or RuleBasedClaimDecomposer()
        self.classifier = classifier or RuleBasedClaimClassifier()
        self.router = router or ToolRouter()
        self.draft_generator = draft_generator or TemplateDraftGenerator()
        self.audit_repo = audit_repo or InMemoryAuditRepository()
        self.module_registry = module_registry or ModuleRegistry()
        # Default to the no-op translator so existing behavior is preserved.
        self.tool_input_translator = (
            tool_input_translator or NullToolInputTranslator()
        )
        # Default to the deterministic lexical detector so behavior is
        # predictable and tests don't drift into LLM calls.
        self.contradiction_detector = (
            contradiction_detector or LexicalContradictionDetector()
        )
        # Empty catalog + null router by default — existing pipelines
        # behave exactly as before. Only when a catalog is supplied does
        # the discovery layer activate.
        self.data_catalog = data_catalog or DataCatalog()
        self.data_source_router = data_source_router or NullDataSourceRouter()
        # source_id -> DataSource. Used by the pipeline to look up the
        # native handle of a connector the discovery layer selected.
        self._data_sources: dict[str, Any] = dict(data_sources or {})

        # Default retriever: an empty in-memory one that can accept per-request docs.
        self.retriever = retriever or LocalDocumentRetriever()

        # Default tool set covers the routes the router declares.
        default_tools: dict[str, Tool] = {
            "calculator": CalculatorTool(),
            "sql_executor": DuckDBSqlExecutor(),
            "python_executor": LocalSubprocessPythonExecutor(),
            "rule_engine": RuleEngineTool(),
            "theorem_prover": TheoremProverStub(),
            "causal_model": CausalModelStub(),
            "ab_test": ABTestCausalTool(),
            "causal_inference": CausalInferenceTool(),
            "forecast": ForecastTool(),
            "optimizer": LinearOptimizerTool(),
        }
        if tools:
            default_tools.update(tools)
        self.evidence_builder = evidence_builder or EvidenceBuilder(
            tools=default_tools, retriever=self.retriever
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, request: PipelineRequest) -> FinalAnswer:
        trace = AuditTrace(original_question=request.question)

        # Per-request retriever: replace the default in-memory store so each
        # call is isolated from prior runs.
        if request.documents:
            self.retriever = LocalDocumentRetriever(documents=list(request.documents))
            self.evidence_builder = EvidenceBuilder(
                tools=self._tool_dict(),
                retriever=self.retriever,
            )

        # 1-3 — decompose + classify
        proposed_claims = self.decomposer.decompose(request.question)
        classified_claims = self.classifier.classify_all(proposed_claims)
        trace.decomposed_claims = classified_claims
        trace.claim_classifications = {
            c.id: c.epistemic_type.value for c in classified_claims
        }

        # 4-5 — domain modules + routing
        # Module enrichment is a no-op when no module is ACTIVE; the registry
        # gates this for us.
        for module in self.module_registry.active():
            classified_claims = module.enrich_claims(classified_claims)
        tasks = self.router.route_all(classified_claims)
        trace.module_versions_used = {
            m.spec.name: m.spec.version for m in self.module_registry.active()
        }

        # 6 — execute tasks (per-request context lets rule engine see policy docs)
        ctx: dict[str, Any] = {
            "documents": [d.model_dump() for d in request.documents],
            **request.extra_context,
        }
        if request.documents:
            # Surface documents as candidate rules so procedural claims hit them.
            ctx["rules"] = [
                {
                    "name": d.name,
                    "text": d.text,
                    "effective_date": d.effective_date,
                }
                for d in request.documents
            ]

        # 6 (pre-translate) — Discovery: pick relevant catalog entries.
        # Recorded to the audit trace and surfaced via ctx["data"] so the
        # LLM-assisted translator can write payloads against them. For
        # warehouse sources we ALSO drop the live native handle into
        # ctx["sql_connection"] so the SQL executor can run queries
        # against the real warehouse instead of an empty default.
        consulted_sources = self.data_source_router.route(
            question=request.question,
            claims=classified_claims,
            catalog=self.data_catalog,
        )
        trace.data_sources_consulted = list(consulted_sources)
        if consulted_sources:
            existing_data = ctx.get("data") if isinstance(ctx.get("data"), dict) else {}
            data_block = dict(existing_data or {})
            for entry in consulted_sources:
                data_block.setdefault(
                    f"source:{entry.source_id}",
                    {
                        "kind": entry.kind.value,
                        "description": entry.description,
                        "tables": [
                            {
                                "name": t.name,
                                "columns": t.columns,
                                "row_count": t.row_count,
                                "sample_rows": t.sample_rows,
                            }
                            for t in entry.tables
                        ],
                    },
                )
            ctx["data"] = data_block

            # Drop warehouse connections into ctx for the SQL executor.
            # First warehouse wins — multi-warehouse routing is future work.
            for entry in consulted_sources:
                handle = self._data_sources.get(entry.source_id)
                if handle is not None and "sql_connection" not in ctx:
                    try:
                        ctx["sql_connection"] = handle.handle()
                    except Exception:
                        pass
                    break

        # 6a — Optional LLM-assisted tool-input translation. The translator
        # only fills in payloads for tools that (a) appear in the routed
        # tasks and (b) the user hasn't already supplied context for. Output
        # is merged into ctx without overwriting user-supplied keys.
        needed_tools: set[str] = {t for task in tasks for t in task.tool_names}
        relevant_specs = [
            TOOL_INPUT_SPECS[name]
            for name in needed_tools
            if name in TOOL_INPUT_SPECS
        ]
        if relevant_specs:
            translated = self.tool_input_translator.translate(
                question=request.question,
                tool_specs=relevant_specs,
                existing_context=ctx,
            )
            for key, value in (translated or {}).items():
                ctx.setdefault(key, value)

        updated_claims, evidence, tool_records = self.evidence_builder.execute(
            tasks, context=ctx
        )
        trace.tools_called = tool_records

        # 7 — contradiction sweep
        evidence, contradictions = check_contradictions(
            evidence, detector=self.contradiction_detector
        )
        trace.retrieved_evidence = evidence
        trace.contradictions_found = contradictions

        # 8-9 — evidence table + verdicts
        verdicts = assign_verdicts(updated_claims, evidence)
        trace.verdicts = verdicts
        table = EvidenceTable(
            original_question=request.question,
            claims=updated_claims,
            evidence=evidence,
            verdicts=verdicts,
        )

        # 9b — SHADOW modules: observe the active path, record their proposed
        # verdicts to the audit trace. SHADOW output NEVER affects ``final``.
        # Failures inside a shadow module are swallowed: they cannot crash the
        # live pipeline. See domain/module_lifecycle.py for the safety boundary.
        trace.shadow_verdicts = self._record_shadow_verdicts(
            updated_claims, evidence, verdicts
        )

        # 10 — draft answer
        draft = self.draft_generator.generate(table)
        trace.draft_answer = draft

        # 11-12 — extract + re-verify draft claims
        extractor = TemplateDraftClaimExtractor()
        draft_claims = extractor.extract(draft, table)
        draft_verdicts = verify_draft_claims(draft_claims, table)

        # 13 — revise into final answer
        final = revise_answer(
            question=request.question,
            draft=draft,
            draft_verdicts=draft_verdicts,
            table=table,
            audit_id=trace.audit_id,
        )
        trace.final_answer = final.answer
        trace.unsupported_claims_removed = list(final.unsupported_or_uncertain_claims)

        # 14 — persist
        self.audit_repo.save(trace)

        # 15 — return
        return final

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tool_dict(self) -> dict[str, Tool]:
        # Reach into the existing evidence_builder's registry so the rebuilt
        # builder shares the same tools.
        return dict(getattr(self.evidence_builder, "_tools", {}))

    def _record_shadow_verdicts(
        self,
        claims: list,
        evidence: list,
        active_verdicts: list,
    ) -> list:
        """Run SHADOW modules over the active path's claims+evidence and
        return the resulting ``ShadowVerdict`` records. Output is recorded
        only — the active verdicts are unchanged."""
        from ..domain.module_lifecycle import ShadowVerdict

        records: list = []
        active_by_claim = {v.claim_id: v for v in active_verdicts}

        for module in self.module_registry.shadow():
            for claim in claims:
                try:
                    proposed = module.validate_evidence(claim, evidence)
                except Exception:
                    # Never let a shadow module crash the live pipeline.
                    continue
                if proposed is None:
                    continue
                active = active_by_claim.get(claim.id)
                agrees = (
                    active is not None
                    and proposed.verdict == active.verdict
                )
                records.append(
                    ShadowVerdict(
                        module_name=module.spec.name,
                        module_version=module.spec.version,
                        claim_id=claim.id,
                        proposed_verdict=proposed.verdict.value,
                        proposed_confidence=proposed.confidence.value,
                        rationale=proposed.rationale,
                        agrees_with_active=agrees,
                    )
                )
        return records
