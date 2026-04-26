"""Tests for the 10-stage ModuleDevelopmentPipeline + promotion gate."""

from __future__ import annotations

import pytest

from factuality_harness.application.module_lifecycle import (
    ModuleDevelopmentPipeline,
    ProposedDomainModule,
    promote_module,
)
from factuality_harness.application.module_registry import ModuleRegistry
from factuality_harness.domain.audit import AuditTrace
from factuality_harness.domain.claims import Claim
from factuality_harness.domain.confidence import ConfidenceLevel
from factuality_harness.domain.epistemic_types import EpistemicType
from factuality_harness.domain.module_lifecycle import (
    PromotionGate,
    SourceProposal,
)
from factuality_harness.domain.modules import ModuleStatus
from factuality_harness.domain.verdicts import ClaimVerdict, Verdict
from factuality_harness.evals.scoring import EvalSummary


# ---------------------------------------------------------------------------
# Stage-by-stage tests
# ---------------------------------------------------------------------------


def _make_trace(question: str, verdicts: list[Verdict]) -> AuditTrace:
    claim_verdicts = [
        ClaimVerdict(
            claim_id=f"claim_{i}",
            verdict=v,
            confidence=ConfidenceLevel.LOW,
            rationale="x",
        )
        for i, v in enumerate(verdicts)
    ]
    claims = [
        Claim(
            id=f"claim_{i}",
            text=question,
            parent_question=question,
            epistemic_type=EpistemicType.UNKNOWN,
        )
        for i, _ in enumerate(verdicts)
    ]
    return AuditTrace(
        original_question=question,
        decomposed_claims=claims,
        verdicts=claim_verdicts,
    )


def test_detect_domain_gap_finds_majority_weak_traces():
    pipe = ModuleDevelopmentPipeline()
    traces = [
        _make_trace("How are quarterly sales trending?", [Verdict.UNSUPPORTED]),
        _make_trace("What was Q1 revenue?", [Verdict.UNCLEAR]),
        _make_trace("What is 2 + 2?", [Verdict.COMPUTED]),
    ]
    gaps = pipe.detect_domain_gap(traces, domain_hint="finance")
    assert len(gaps) == 1
    assert gaps[0].name == "finance"
    assert gaps[0].severity > 0.5
    assert "quarterly sales" in " ".join(gaps[0].representative_queries).lower()


def test_detect_domain_gap_returns_empty_when_no_weak_traces():
    pipe = ModuleDevelopmentPipeline()
    traces = [_make_trace("x", [Verdict.SUPPORTED])]
    assert pipe.detect_domain_gap(traces) == []


def test_propose_taxonomy_groups_by_epistemic_type():
    pipe = ModuleDevelopmentPipeline()
    queries = [
        "What is the percentage increase from 100 to 125?",  # NUMERICAL
        "What is 2 + 2?",  # NUMERICAL
        "Did the campaign cause sales to increase?",  # CAUSAL
        "Will demand grow next quarter?",  # PREDICTIVE
    ]
    taxonomy = pipe.propose_claim_taxonomy(queries)
    types = {c.epistemic_type for c in taxonomy}
    assert EpistemicType.NUMERICAL in types
    assert EpistemicType.CAUSAL in types
    assert EpistemicType.PREDICTIVE in types
    # NUMERICAL category should aggregate both numerical queries.
    numeric = next(c for c in taxonomy if c.epistemic_type == EpistemicType.NUMERICAL)
    assert len(numeric.example_queries) == 2


def test_identify_authoritative_sources_marks_existing():
    pipe = ModuleDevelopmentPipeline()
    taxonomy = pipe.propose_claim_taxonomy(
        ["What is the percentage increase from 100 to 125?"]
    )
    sources = pipe.identify_authoritative_sources(taxonomy)
    # Every default tool for NUMERICAL exists; all should be is_existing=True.
    assert all(s.is_existing for s in sources)
    assert any(s.source_id == "calculator" for s in sources)


def test_propose_routes_returns_one_per_category():
    pipe = ModuleDevelopmentPipeline()
    taxonomy = pipe.propose_claim_taxonomy(
        [
            "What is 2 + 2?",
            "Did X cause Y?",
        ]
    )
    routes = pipe.propose_tool_routes(taxonomy)
    assert {r.claim_category for r in routes} == {c.name for c in taxonomy}


def test_propose_validation_rules_reflects_calibrator_expectations():
    pipe = ModuleDevelopmentPipeline()
    taxonomy = pipe.propose_claim_taxonomy(
        ["Did the email cause more conversions?"]
    )
    rules = pipe.propose_validation_rules(taxonomy)
    causal = next(
        r for r in rules if r.claim_type == EpistemicType.CAUSAL
    )
    assert causal.requires_human_review is True
    assert causal.requires_contradiction_search is True


def test_generate_benchmark_cases_forbids_verified_on_predictive():
    pipe = ModuleDevelopmentPipeline()
    queries = ["Will revenue grow next quarter?"]
    taxonomy = pipe.propose_claim_taxonomy(queries)
    cases = pipe.generate_benchmark_cases(
        taxonomy, queries, domain_name="forecast"
    )
    assert len(cases) == 1
    assert "VERIFIED" in cases[0].expected.verdicts_must_not_include


# ---------------------------------------------------------------------------
# Develop-from-queries: full stages 1-7 in one call
# ---------------------------------------------------------------------------


def test_develop_from_queries_produces_deployable_proposal():
    pipe = ModuleDevelopmentPipeline()
    proposal = pipe.develop_from_queries(
        domain_name="finance",
        example_queries=[
            "What was Q1 revenue?",  # NUMERICAL via "what is"
            "What is the percentage increase from 100 to 125?",
            "Did the marketing campaign cause more conversions?",
        ],
    )
    assert proposal.spec.name == "finance"
    assert proposal.spec.status == ModuleStatus.DRAFT
    assert proposal.taxonomy
    assert proposal.routes
    assert proposal.benchmark_case_names
    assert proposal.is_deployable, (
        "proposal should be deployable since every default source is existing"
    )


def test_develop_from_queries_uses_prior_audit_traces_when_provided():
    pipe = ModuleDevelopmentPipeline()
    traces = [
        _make_trace("What was Q1 revenue?", [Verdict.UNSUPPORTED]),
        _make_trace("What was Q2 revenue?", [Verdict.UNCLEAR]),
    ]
    proposal = pipe.develop_from_queries(
        domain_name="finance",
        example_queries=["What is the EBITDA margin?"],
        prior_audit_traces=traces,
    )
    # Gap should reflect the prior trace count, not just the seed queries.
    assert proposal.gap.severity > 0
    assert any(
        "Q1 revenue" in q for q in proposal.gap.representative_queries
    )


# ---------------------------------------------------------------------------
# Deploy as SHADOW
# ---------------------------------------------------------------------------


def test_deploy_shadow_mode_registers_module():
    pipe = ModuleDevelopmentPipeline()
    proposal = pipe.develop_from_queries(
        domain_name="finance",
        example_queries=["What is the percentage increase from 100 to 125?"],
    )
    registry = ModuleRegistry()
    module = pipe.deploy_shadow_mode(proposal, registry)
    assert isinstance(module, ProposedDomainModule)
    assert module.spec.status == ModuleStatus.SHADOW
    # And the registry now has it in its shadow set.
    shadow_modules = registry.shadow()
    assert any(m.spec.name == "finance" for m in shadow_modules)


def test_deploy_shadow_mode_refuses_non_deployable_proposal():
    pipe = ModuleDevelopmentPipeline()
    proposal = pipe.develop_from_queries(
        domain_name="finance",
        example_queries=["What is the percentage increase from 100 to 125?"],
    )
    # Replace one source with a fake one to flip is_deployable to False.
    proposal = proposal.model_copy(
        update={
            "sources": proposal.sources
            + [
                SourceProposal(
                    source_id="nonexistent_warehouse",
                    description="x",
                    tool_kind="tool",
                    is_existing=False,
                )
            ]
        }
    )
    assert proposal.is_deployable is False
    with pytest.raises(ValueError, match="not deployable"):
        pipe.deploy_shadow_mode(proposal, ModuleRegistry())


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------


def _summary(metrics: dict[str, float]) -> EvalSummary:
    return EvalSummary(
        total_cases=10,
        passed_cases=int(metrics.get("case_pass_rate", 0) * 10),
        failed_cases=10 - int(metrics.get("case_pass_rate", 0) * 10),
        case_pass_rate=metrics.get("case_pass_rate", 0),
        metrics=metrics,
        cases=[],
    )


def test_evaluate_promotion_blocks_when_eval_fails():
    pipe = ModuleDevelopmentPipeline()
    decision = pipe.evaluate_promotion(
        module_name="finance",
        module_version="0.0.1",
        eval_summary=_summary(
            {
                "case_pass_rate": 0.7,
                "classification_accuracy": 0.9,
                "overclaim_rate": 0.0,
            }
        ),
        shadow_run_count=100,
        shadow_agreement_rate=0.95,
    )
    assert decision.eligible is False
    assert any("case_pass_rate" in r for r in decision.reasons)


def test_evaluate_promotion_blocks_on_overclaim():
    pipe = ModuleDevelopmentPipeline()
    decision = pipe.evaluate_promotion(
        module_name="finance",
        module_version="0.0.1",
        eval_summary=_summary(
            {
                "case_pass_rate": 0.95,
                "classification_accuracy": 0.95,
                "overclaim_rate": 0.2,  # too high
            }
        ),
        shadow_run_count=100,
        shadow_agreement_rate=0.95,
    )
    assert decision.eligible is False
    assert any("overclaim_rate" in r for r in decision.reasons)


def test_evaluate_promotion_blocks_when_shadow_runs_too_few():
    pipe = ModuleDevelopmentPipeline()
    decision = pipe.evaluate_promotion(
        module_name="finance",
        module_version="0.0.1",
        eval_summary=_summary(
            {
                "case_pass_rate": 1.0,
                "classification_accuracy": 1.0,
                "overclaim_rate": 0.0,
            }
        ),
        shadow_run_count=5,  # below default min_shadow_runs=20
        shadow_agreement_rate=1.0,
    )
    assert decision.eligible is False
    assert any("shadow runs" in r for r in decision.reasons)


def test_evaluate_promotion_eligible_when_thresholds_met():
    pipe = ModuleDevelopmentPipeline()
    decision = pipe.evaluate_promotion(
        module_name="finance",
        module_version="0.0.1",
        eval_summary=_summary(
            {
                "case_pass_rate": 1.0,
                "classification_accuracy": 1.0,
                "overclaim_rate": 0.0,
            }
        ),
        shadow_run_count=50,
        shadow_agreement_rate=0.9,
    )
    assert decision.eligible is True
    assert decision.requires_human_approval is True
    assert decision.promoted is False  # never auto-promotes


# ---------------------------------------------------------------------------
# promote_module: the only sanctioned write path
# ---------------------------------------------------------------------------


def _setup_shadow_module() -> tuple[ModuleRegistry, str]:
    pipe = ModuleDevelopmentPipeline()
    proposal = pipe.develop_from_queries(
        domain_name="finance_test",
        example_queries=["What is the percentage increase from 100 to 125?"],
    )
    registry = ModuleRegistry()
    pipe.deploy_shadow_mode(proposal, registry)
    return registry, "finance_test"


def test_promote_module_refuses_when_not_eligible():
    registry, name = _setup_shadow_module()
    bad_decision = ModuleDevelopmentPipeline().evaluate_promotion(
        module_name=name,
        module_version="0.0.1",
        eval_summary=_summary({"case_pass_rate": 0.0}),
    )
    result = promote_module(name, registry, decision=bad_decision, human_approval=True)
    assert result.promoted is False
    assert registry.get(name).spec.status == ModuleStatus.SHADOW


def test_promote_module_refuses_without_human_approval():
    registry, name = _setup_shadow_module()
    good = ModuleDevelopmentPipeline().evaluate_promotion(
        module_name=name,
        module_version="0.0.1",
        eval_summary=_summary(
            {
                "case_pass_rate": 1.0,
                "classification_accuracy": 1.0,
                "overclaim_rate": 0.0,
            }
        ),
        shadow_run_count=50,
        shadow_agreement_rate=0.9,
    )
    assert good.eligible is True
    # No human approval supplied -> module stays SHADOW.
    result = promote_module(name, registry, decision=good, human_approval=False)
    assert result.promoted is False
    assert registry.get(name).spec.status == ModuleStatus.SHADOW


def test_promote_module_flips_status_with_human_approval():
    registry, name = _setup_shadow_module()
    good = ModuleDevelopmentPipeline().evaluate_promotion(
        module_name=name,
        module_version="0.0.1",
        eval_summary=_summary(
            {
                "case_pass_rate": 1.0,
                "classification_accuracy": 1.0,
                "overclaim_rate": 0.0,
            }
        ),
        shadow_run_count=50,
        shadow_agreement_rate=0.9,
    )
    result = promote_module(name, registry, decision=good, human_approval=True)
    assert result.promoted is True
    assert registry.get(name).spec.status == ModuleStatus.ACTIVE


# ---------------------------------------------------------------------------
# ProposedDomainModule applies_to heuristic
# ---------------------------------------------------------------------------


def test_proposed_module_applies_to_matches_taxonomy_keywords():
    pipe = ModuleDevelopmentPipeline()
    proposal = pipe.develop_from_queries(
        domain_name="finance",
        example_queries=[
            "What was Q1 revenue?",
            "What is the EBITDA margin?",
            "What is the percentage increase from 100 to 125?",
        ],
    )
    module = ProposedDomainModule(proposal)
    score = module.applies_to(
        "What was the revenue trend last quarter?", []
    )
    assert score > 0.0
