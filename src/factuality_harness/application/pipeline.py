"""End-to-end factuality pipeline.

This is the orchestrator. It is intentionally explicit — every step is a method
call that produces inspectable state. It does not call LLM APIs or external
services directly; those live behind ``infrastructure`` ports passed in via
construction.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

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
from ..infrastructure.tools.base import Tool
from ..infrastructure.tools.calculator import CalculatorTool
from ..infrastructure.tools.causal_model_stub import CausalModelStub
from ..infrastructure.tools.optimizer_stub import OptimizerStub
from ..infrastructure.tools.rule_engine import RuleEngineTool
from ..infrastructure.tools.theorem_prover_stub import TheoremProverStub
from .claim_classifier import ClaimClassifier, RuleBasedClaimClassifier
from .claim_decomposer import ClaimDecomposer, RuleBasedClaimDecomposer
from .contradiction_checker import check_contradictions
from .draft_generator import DraftGenerator, TemplateDraftGenerator
from .evidence_builder import EvidenceBuilder
from .final_verifier import (
    TemplateDraftClaimExtractor,
    revise_answer,
    verify_draft_claims,
)
from .module_registry import ModuleRegistry
from .router import ToolRouter
from .uncertainty_calibrator import assign_verdicts


class PipelineRequest(BaseModel):
    question: str
    domain_hint: str | None = None
    documents: list[Document] = []


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
    ) -> None:
        self.decomposer = decomposer or RuleBasedClaimDecomposer()
        self.classifier = classifier or RuleBasedClaimClassifier()
        self.router = router or ToolRouter()
        self.draft_generator = draft_generator or TemplateDraftGenerator()
        self.audit_repo = audit_repo or InMemoryAuditRepository()
        self.module_registry = module_registry or ModuleRegistry()

        # Default retriever: an empty in-memory one that can accept per-request docs.
        self.retriever = retriever or LocalDocumentRetriever()

        # Default tool set covers the routes the router declares.
        default_tools: dict[str, Tool] = {
            "calculator": CalculatorTool(),
            "rule_engine": RuleEngineTool(),
            "theorem_prover": TheoremProverStub(),
            "causal_model": CausalModelStub(),
            "optimizer": OptimizerStub(),
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
        ctx: dict[str, Any] = {"documents": [d.model_dump() for d in request.documents]}
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

        updated_claims, evidence, tool_records = self.evidence_builder.execute(
            tasks, context=ctx
        )
        trace.tools_called = tool_records

        # 7 — contradiction sweep
        evidence, contradictions = check_contradictions(evidence)
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
