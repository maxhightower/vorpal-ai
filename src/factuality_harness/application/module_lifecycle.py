"""Module-development lifecycle: the 10-stage pipeline.

The ``ModuleDevelopmentPipeline`` orchestrator turns a domain gap (or a
manually-supplied list of example queries) into a vetted, deployable
module proposal. Stages are deterministic by default — they use the
existing rule-based decomposer/classifier, the active eval runner, and
the registry already in the harness — so the lifecycle works without an
LLM. An LLM-backed taxonomy / route / validation proposer can be wired
in behind the same protocol if higher-quality proposals are wanted.

Safety boundaries enforced here, per the original spec:

 - Shadow modules are registered with status SHADOW only after passing
   ``ModuleProposal.is_deployable`` (every proposed source exists).
 - ``evaluate_promotion`` evaluates whether a module is *eligible* for
   ACTIVE; it never flips the registry status by itself.
 - ``promote_module`` is the only place in the codebase that flips a
   module from SHADOW to ACTIVE, and it refuses without explicit
   human approval whenever the decision says it's required (which is
   the default).

Stages mirror the spec's 10-stage list verbatim:

  1. detect_domain_gap
  2. collect_representative_queries
  3. propose_claim_taxonomy
  4. identify_authoritative_sources
  5. propose_tool_routes
  6. propose_validation_rules
  7. generate_benchmark_cases
  8. run_module_evals
  9. deploy_shadow_mode
 10. promote_if_passes_thresholds
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ..domain.audit import AuditTrace
from ..domain.epistemic_types import EpistemicType
from ..domain.evidence import EvidenceRequirement, SourceQuality, SourceType
from ..domain.module_lifecycle import (
    ClaimCategory,
    DomainGap,
    ModuleProposal,
    PromotionDecision,
    PromotionGate,
    RouteProposal,
    SourceProposal,
)
from ..domain.modules import ModuleSpec, ModuleStatus
from ..evals.cases import EvalCase, ExpectedOutcome
from ..evals.scoring import EvalSummary
from ..modules.base import BaseDomainModule
from .module_registry import ModuleRegistry
from .router import ToolRouter


# ---------------------------------------------------------------------------
# Tool / source defaults per epistemic type
# ---------------------------------------------------------------------------


# What tool a proposed module should ask the router for, per claim type.
# Keys overlap with ``application/router.py::_DEFAULT_ROUTES``; we keep them
# duplicated here so a proposal can be reasoned about without booting the
# pipeline.
_DEFAULT_TOOLS_BY_TYPE: dict[EpistemicType, list[str]] = {
    EpistemicType.NUMERICAL: ["calculator", "sql_executor", "python_executor"],
    EpistemicType.DIRECT_FACT: ["local_document_retriever", "sql_executor"],
    EpistemicType.LOGICAL: ["rule_engine", "theorem_prover", "python_executor"],
    EpistemicType.PROCEDURAL: ["rule_engine", "local_document_retriever"],
    EpistemicType.CAUSAL: ["ab_test", "causal_inference", "causal_model"],
    EpistemicType.PREDICTIVE: ["forecast", "causal_model"],
    EpistemicType.OPTIMIZATION: ["optimizer", "python_executor"],
    EpistemicType.INTERPRETIVE: ["local_document_retriever"],
    EpistemicType.SPECULATIVE: [],
    EpistemicType.UNKNOWN: ["local_document_retriever"],
}

# Existing tool names — proposals that reference anything outside this set
# are flagged as non-deployable.
_EXISTING_TOOLS: set[str] = {
    "calculator",
    "sql_executor",
    "python_executor",
    "rule_engine",
    "theorem_prover",
    "causal_model",
    "ab_test",
    "causal_inference",
    "forecast",
    "optimizer",
    "local_document_retriever",
}

_TYPE_QUALITY: dict[EpistemicType, SourceQuality] = {
    EpistemicType.NUMERICAL: SourceQuality.AUTHORITATIVE,
    EpistemicType.DIRECT_FACT: SourceQuality.SECONDARY,
    EpistemicType.LOGICAL: SourceQuality.AUTHORITATIVE,
    EpistemicType.PROCEDURAL: SourceQuality.AUTHORITATIVE,
    EpistemicType.CAUSAL: SourceQuality.SECONDARY,
    EpistemicType.PREDICTIVE: SourceQuality.SECONDARY,
    EpistemicType.OPTIMIZATION: SourceQuality.AUTHORITATIVE,
    EpistemicType.INTERPRETIVE: SourceQuality.SECONDARY,
    EpistemicType.SPECULATIVE: SourceQuality.UNKNOWN,
    EpistemicType.UNKNOWN: SourceQuality.SECONDARY,
}


# ---------------------------------------------------------------------------
# ProposedDomainModule: a generic ``BaseDomainModule`` driven by a proposal
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")


def _keywords(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 2}


class ProposedDomainModule(BaseDomainModule):
    """A ``BaseDomainModule`` whose behavior is driven entirely by a
    ``ModuleProposal``. Useful for shadow / draft deployment without
    hand-writing a Python class.

    ``applies_to`` is a keyword-overlap heuristic against the taxonomy.
    ``required_evidence_for`` returns the proposal's per-type requirements.
    ``validate_evidence`` returns ``None`` — proposed modules observe the
    active path; they do not override its verdicts. Shadow recording uses
    the active verdict for comparison.
    """

    def __init__(self, proposal: ModuleProposal) -> None:
        super().__init__(proposal.spec)
        self.proposal = proposal
        self._taxonomy_keywords = {
            cat.name: _keywords(
                " ".join([cat.description, *cat.example_queries])
            )
            for cat in proposal.taxonomy
        }
        self._req_by_type: dict[EpistemicType, list[EvidenceRequirement]] = {}
        for req in proposal.evidence_requirements:
            self._req_by_type.setdefault(req.claim_type, []).append(req)

    def applies_to(self, question: str, claims: list) -> float:
        if not self._taxonomy_keywords:
            return 0.0
        question_kw = _keywords(question)
        claim_kw: set[str] = set()
        for c in claims:
            claim_kw |= _keywords(c.text)
        text_kw = question_kw | claim_kw
        if not text_kw:
            return 0.0
        scores: list[float] = []
        for kw in self._taxonomy_keywords.values():
            if not kw:
                continue
            scores.append(len(text_kw & kw) / len(kw))
        return max(scores) if scores else 0.0

    def required_evidence_for(self, claim) -> list[EvidenceRequirement]:
        return list(self._req_by_type.get(claim.epistemic_type, []))


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


class ModuleDevelopmentPipeline:
    """Orchestrates the 10 lifecycle stages.

    Each stage is a method so callers can run them independently or
    compose their own flow. ``develop_from_queries`` runs the full chain
    end-to-end.
    """

    def __init__(
        self,
        *,
        router: ToolRouter | None = None,
    ) -> None:
        self._router = router or ToolRouter()

    # ------------------------------------------------------------------
    # 1. Detect a domain gap from prior audit traces
    # ------------------------------------------------------------------

    def detect_domain_gap(
        self,
        audit_traces: list[AuditTrace],
        *,
        domain_hint: str | None = None,
        unsupported_threshold: float = 0.5,
    ) -> list[DomainGap]:
        """Identify questions whose verdicts are mostly UNSUPPORTED/UNCLEAR.

        The default heuristic groups all such questions under a single
        gap labelled ``domain_hint`` (or ``"general"`` if none is given).
        A more sophisticated implementation could cluster by topic; for
        the MVP this surfaces the gap shape without inventing topics.
        """
        weak_verdicts = {"UNSUPPORTED", "UNCLEAR"}
        bad_traces: list[AuditTrace] = []
        for t in audit_traces:
            if not t.verdicts:
                continue
            weak = sum(1 for v in t.verdicts if v.verdict.value in weak_verdicts)
            if weak / len(t.verdicts) >= unsupported_threshold:
                bad_traces.append(t)

        if not bad_traces:
            return []

        gap_name = (domain_hint or "general").strip().lower().replace(" ", "_")
        return [
            DomainGap(
                name=gap_name,
                description=(
                    f"{len(bad_traces)} of {len(audit_traces)} prior runs "
                    f"produced majority-weak verdicts."
                ),
                representative_audit_ids=[t.audit_id for t in bad_traces[:25]],
                representative_queries=[t.original_question for t in bad_traces[:25]],
                severity=len(bad_traces) / max(len(audit_traces), 1),
            )
        ]

    # ------------------------------------------------------------------
    # 2. Collect representative queries (already on the gap; pass-through)
    # ------------------------------------------------------------------

    def collect_representative_queries(
        self,
        gap: DomainGap,
        *,
        extra_queries: list[str] | None = None,
        max_queries: int = 50,
    ) -> list[str]:
        merged = list(gap.representative_queries)
        for q in extra_queries or []:
            if q and q not in merged:
                merged.append(q)
        return merged[:max_queries]

    # ------------------------------------------------------------------
    # 3. Propose a claim taxonomy from the queries
    # ------------------------------------------------------------------

    def propose_claim_taxonomy(self, queries: list[str]) -> list[ClaimCategory]:
        """Run the existing rule-based classifier on each query, then
        aggregate by epistemic type into one category per type.

        Each category's ``name`` is ``<lowercase_type>_query`` and its
        ``example_queries`` are the input queries that classified into it.
        Categories with zero queries are dropped.
        """
        # Local imports keep module-import cost low.
        from .claim_classifier import RuleBasedClaimClassifier
        from .claim_decomposer import RuleBasedClaimDecomposer

        decomposer = RuleBasedClaimDecomposer()
        classifier = RuleBasedClaimClassifier()

        type_counts: Counter[EpistemicType] = Counter()
        examples_by_type: dict[EpistemicType, list[str]] = {}
        for q in queries:
            claims = decomposer.decompose(q)
            classified = classifier.classify_all(claims)
            if not classified:
                continue
            etype = classified[0].epistemic_type
            type_counts[etype] += 1
            examples_by_type.setdefault(etype, []).append(q)

        categories: list[ClaimCategory] = []
        for etype, count in type_counts.most_common():
            categories.append(
                ClaimCategory(
                    name=f"{etype.value.lower()}_query",
                    description=(
                        f"{count} representative {etype.value} claim(s) "
                        "observed in the gap."
                    ),
                    epistemic_type=etype,
                    example_queries=examples_by_type[etype][:5],
                )
            )
        return categories

    # ------------------------------------------------------------------
    # 4. Identify authoritative sources
    # ------------------------------------------------------------------

    def identify_authoritative_sources(
        self, taxonomy: list[ClaimCategory]
    ) -> list[SourceProposal]:
        """Map each claim category to the existing tools the harness would
        use for that epistemic type. Sources that aren't in
        ``_EXISTING_TOOLS`` are still returned but flagged
        ``is_existing=False``."""
        proposed: dict[str, SourceProposal] = {}
        for cat in taxonomy:
            for tool_name in _DEFAULT_TOOLS_BY_TYPE.get(cat.epistemic_type, []):
                if tool_name in proposed:
                    continue
                proposed[tool_name] = SourceProposal(
                    source_id=tool_name,
                    description=(
                        f"Tool selected for {cat.epistemic_type.value} claims."
                    ),
                    tool_kind="tool",
                    is_existing=tool_name in _EXISTING_TOOLS,
                )
        return list(proposed.values())

    # ------------------------------------------------------------------
    # 5. Propose tool routes
    # ------------------------------------------------------------------

    def propose_tool_routes(
        self, taxonomy: list[ClaimCategory]
    ) -> list[RouteProposal]:
        return [
            RouteProposal(
                claim_category=cat.name,
                tool_names=list(_DEFAULT_TOOLS_BY_TYPE.get(cat.epistemic_type, [])),
            )
            for cat in taxonomy
        ]

    # ------------------------------------------------------------------
    # 6. Propose validation rules
    # ------------------------------------------------------------------

    def propose_validation_rules(
        self, taxonomy: list[ClaimCategory]
    ) -> list[EvidenceRequirement]:
        """Per-type evidence requirements aligned with the harness's
        existing verdict rules — ensures shadow modules don't overclaim."""
        out: list[EvidenceRequirement] = []
        for cat in taxonomy:
            etype = cat.epistemic_type
            req = EvidenceRequirement(
                claim_type=etype,
                required_source_types=_required_source_types_for(etype),
                minimum_source_quality=_TYPE_QUALITY.get(
                    etype, SourceQuality.SECONDARY
                ),
                requires_date_check=etype
                in (EpistemicType.DIRECT_FACT, EpistemicType.PROCEDURAL),
                requires_computation=etype == EpistemicType.NUMERICAL,
                requires_contradiction_search=True,
                requires_human_review=etype == EpistemicType.CAUSAL,
                rationale=(
                    f"Default requirement for {etype.value} claims; matches "
                    "the harness verdict calibrator's expectations."
                ),
            )
            out.append(req)
        return out

    # ------------------------------------------------------------------
    # 7. Generate benchmark cases
    # ------------------------------------------------------------------

    def generate_benchmark_cases(
        self,
        taxonomy: list[ClaimCategory],
        queries: list[str],
        *,
        domain_name: str,
    ) -> list[EvalCase]:
        """Build one ``EvalCase`` per representative query.

        Expectations are intentionally conservative: assert the claim's
        epistemic type, and forbid VERIFIED on PREDICTIVE/SPECULATIVE.
        Stronger expectations require ground truth we don't have at
        proposal time.
        """
        from .claim_classifier import RuleBasedClaimClassifier
        from .claim_decomposer import RuleBasedClaimDecomposer

        decomposer = RuleBasedClaimDecomposer()
        classifier = RuleBasedClaimClassifier()

        cases: list[EvalCase] = []
        for i, q in enumerate(queries):
            claims = decomposer.decompose(q)
            classified = classifier.classify_all(claims)
            if not classified:
                continue
            etype = classified[0].epistemic_type
            forbid: list[str] = []
            if etype == EpistemicType.PREDICTIVE:
                forbid.append("VERIFIED")
            if etype == EpistemicType.SPECULATIVE:
                forbid.extend(["VERIFIED", "SUPPORTED"])

            cases.append(
                EvalCase(
                    name=f"{domain_name}_proposed_{i:03d}",
                    question=q,
                    expected=ExpectedOutcome(
                        first_claim_type=etype.value,
                        verdicts_must_not_include=forbid,
                    ),
                )
            )
        return cases

    # ------------------------------------------------------------------
    # 8. Run module evals
    # ------------------------------------------------------------------

    def run_module_evals(
        self,
        cases: list[EvalCase],
        pipeline,
    ) -> EvalSummary:
        """Run the proposal's benchmark cases against the supplied
        pipeline and return the eval summary."""
        from ..evals.run_evals import run_eval_cases  # local import

        # Use the same runner as ``fh evals run`` so metrics are comparable.
        # ``run_eval_cases`` builds its own pipeline; for unit-testability
        # we accept a pipeline arg and run cases inline if provided.
        if pipeline is None:
            return run_eval_cases(cases)

        from ..evals.scoring import aggregate, score_case
        from ..application.pipeline import PipelineRequest
        from ..infrastructure.retrieval.base import Document

        results = []
        for case in cases:
            try:
                final = pipeline.run(
                    PipelineRequest(
                        question=case.question,
                        documents=[
                            Document.model_validate(d) for d in case.documents
                        ],
                        extra_context=case.extra_context,
                    )
                )
                trace = pipeline.audit_repo.get(final.audit_id)
                results.append(score_case(case, final, trace))
            except Exception as e:
                results.append(
                    score_case(case, None, None, error=f"{type(e).__name__}: {e}")
                )
        return aggregate(results)

    # ------------------------------------------------------------------
    # End-to-end orchestration (stages 1-7 in one call)
    # ------------------------------------------------------------------

    def develop_from_queries(
        self,
        *,
        domain_name: str,
        example_queries: list[str],
        description: str | None = None,
        prior_audit_traces: list[AuditTrace] | None = None,
    ) -> ModuleProposal:
        """Run stages 1-7 to produce a ``ModuleProposal``. Stages 8-10
        (run evals, deploy as shadow, promote) require a live pipeline
        and registry, so they're called separately."""
        gap = self._gap_from_inputs(
            domain_name=domain_name,
            example_queries=example_queries,
            prior_audit_traces=prior_audit_traces or [],
        )
        queries = self.collect_representative_queries(gap)
        taxonomy = self.propose_claim_taxonomy(queries)
        sources = self.identify_authoritative_sources(taxonomy)
        routes = self.propose_tool_routes(taxonomy)
        rules = self.propose_validation_rules(taxonomy)
        cases = self.generate_benchmark_cases(
            taxonomy, queries, domain_name=domain_name
        )

        spec = ModuleSpec(
            name=domain_name,
            description=description
            or f"Auto-proposed module for the {domain_name!r} domain.",
            version="0.0.1",
            status=ModuleStatus.DRAFT,
            claim_types=[c.epistemic_type.value for c in taxonomy],
            source_policy={"existing_only": True},
            tool_policy={r.claim_category: r.tool_names for r in routes},
            evidence_standards={
                cat.epistemic_type.value: cat.description for cat in taxonomy
            },
            benchmark_cases=[c.model_dump() for c in cases],
        )
        return ModuleProposal(
            spec=spec,
            gap=gap,
            taxonomy=taxonomy,
            sources=sources,
            routes=routes,
            evidence_requirements=rules,
            benchmark_case_names=[c.name for c in cases],
        )

    def _gap_from_inputs(
        self,
        *,
        domain_name: str,
        example_queries: list[str],
        prior_audit_traces: list[AuditTrace],
    ) -> DomainGap:
        """Use ``detect_domain_gap`` if traces are available; otherwise
        synthesize a minimal gap from the supplied queries so callers
        can drive the pipeline without prior audit data."""
        if prior_audit_traces:
            gaps = self.detect_domain_gap(
                prior_audit_traces, domain_hint=domain_name
            )
            if gaps:
                merged_queries = list(gaps[0].representative_queries)
                for q in example_queries:
                    if q not in merged_queries:
                        merged_queries.append(q)
                return gaps[0].model_copy(
                    update={"representative_queries": merged_queries}
                )
        return DomainGap(
            name=domain_name.lower().replace(" ", "_"),
            description=(
                f"Manual proposal for {domain_name}; no prior audit traces."
            ),
            representative_queries=list(example_queries),
            severity=0.0,
        )

    # ------------------------------------------------------------------
    # 9. Deploy as SHADOW
    # ------------------------------------------------------------------

    def deploy_shadow_mode(
        self,
        proposal: ModuleProposal,
        registry: ModuleRegistry,
    ) -> ProposedDomainModule:
        """Register the proposal in SHADOW status, after refusing if any
        proposed source is non-existent. Returns the deployed module."""
        if not proposal.is_deployable:
            missing = [s.source_id for s in proposal.sources if not s.is_existing]
            raise ValueError(
                "ModuleProposal is not deployable. "
                f"Missing source(s): {missing or '<no sources at all>'}"
            )

        spec = proposal.spec.model_copy(update={"status": ModuleStatus.SHADOW})
        proposal_with_shadow_spec = proposal.model_copy(update={"spec": spec})
        module = ProposedDomainModule(proposal_with_shadow_spec)
        registry.register(module)
        return module

    # ------------------------------------------------------------------
    # 10. Promote-if-passes (eligibility check; flip is in promote_module)
    # ------------------------------------------------------------------

    def evaluate_promotion(
        self,
        *,
        module_name: str,
        module_version: str,
        eval_summary: EvalSummary,
        gate: PromotionGate | None = None,
        shadow_run_count: int = 0,
        shadow_agreement_rate: float = 1.0,
    ) -> PromotionDecision:
        """Decide whether the module is *eligible* for ACTIVE.

        This method NEVER changes registry status. Even when
        ``eligible == True``, ``requires_human_approval`` still defaults
        to True and ``promoted`` stays False. The actual flip happens in
        ``promote_module``.
        """
        gate = gate or PromotionGate()
        reasons: list[str] = []

        eval_pass_rate = eval_summary.metrics.get("case_pass_rate", 0.0)
        if eval_pass_rate < gate.min_eval_pass_rate:
            reasons.append(
                f"case_pass_rate {eval_pass_rate:.3f} < {gate.min_eval_pass_rate:.3f}."
            )

        overclaim = eval_summary.metrics.get("overclaim_rate", 0.0)
        if overclaim > gate.max_overclaim_rate:
            reasons.append(
                f"overclaim_rate {overclaim:.3f} > {gate.max_overclaim_rate:.3f}."
            )

        classification = eval_summary.metrics.get("classification_accuracy", 0.0)
        if classification < gate.min_classification_accuracy:
            reasons.append(
                f"classification_accuracy {classification:.3f} < "
                f"{gate.min_classification_accuracy:.3f}."
            )

        if shadow_run_count < gate.min_shadow_runs:
            reasons.append(
                f"shadow runs {shadow_run_count} < {gate.min_shadow_runs}."
            )
        if shadow_agreement_rate < gate.min_shadow_agreement_rate:
            reasons.append(
                f"shadow agreement rate {shadow_agreement_rate:.3f} < "
                f"{gate.min_shadow_agreement_rate:.3f}."
            )

        eligible = not reasons
        if eligible:
            reasons.append("All gate thresholds met.")

        return PromotionDecision(
            module_name=module_name,
            module_version=module_version,
            eligible=eligible,
            reasons=reasons,
            requires_human_approval=True,
            promoted=False,
        )


# ---------------------------------------------------------------------------
# The single sanctioned promotion executor
# ---------------------------------------------------------------------------


def promote_module(
    module_name: str,
    registry: ModuleRegistry,
    *,
    decision: PromotionDecision,
    human_approval: bool = False,
) -> PromotionDecision:
    """Flip a SHADOW module to ACTIVE if (and only if) the gate decided
    it was eligible AND human approval is supplied (or the decision
    explicitly waived it).

    This function is the single sanctioned write path for promotion.
    Calling ``registry.promote(name, ACTIVE)`` directly is a bug.
    """
    if not decision.eligible:
        return decision
    if decision.requires_human_approval and not human_approval:
        return decision
    registry.promote(module_name, ModuleStatus.ACTIVE)
    return decision.model_copy(update={"promoted": True})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _required_source_types_for(etype: EpistemicType) -> list[SourceType]:
    """Map an epistemic type to the harness's expected evidence source
    types. Mirrors the verdict calibrator's per-type rules so that
    ``EvidenceRequirement`` proposals don't drift from what the
    calibrator actually checks."""
    if etype == EpistemicType.NUMERICAL:
        return [SourceType.COMPUTATION, SourceType.STRUCTURED_DATA]
    if etype == EpistemicType.LOGICAL:
        return [SourceType.THEOREM_PROVER, SourceType.RULE_ENGINE]
    if etype == EpistemicType.PROCEDURAL:
        return [SourceType.RULE_ENGINE, SourceType.LOCAL_DOCUMENT]
    if etype == EpistemicType.CAUSAL:
        return [SourceType.CAUSAL_MODEL]
    if etype == EpistemicType.PREDICTIVE:
        return [SourceType.FORECAST_MODEL, SourceType.CAUSAL_MODEL]
    if etype == EpistemicType.OPTIMIZATION:
        return [SourceType.OPTIMIZER]
    if etype == EpistemicType.DIRECT_FACT:
        return [
            SourceType.LOCAL_DOCUMENT,
            SourceType.WEB_PAGE,
            SourceType.STRUCTURED_DATA,
        ]
    if etype == EpistemicType.INTERPRETIVE:
        return [SourceType.LOCAL_DOCUMENT, SourceType.WEB_PAGE]
    return []
