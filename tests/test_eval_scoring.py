"""Per-case scoring + aggregation tests with hand-built audit traces.

We deliberately don't run the pipeline here — these tests pin down what
the scorer treats as a pass vs a regression so the eval harness itself
won't drift on us.
"""

from __future__ import annotations

from datetime import datetime, timezone

from factuality_harness.domain.audit import AuditTrace
from factuality_harness.domain.claims import Claim
from factuality_harness.domain.confidence import ConfidenceLevel
from factuality_harness.domain.epistemic_types import EpistemicType
from factuality_harness.domain.evidence import (
    Evidence,
    EvidenceTable,
    SourceQuality,
    SourceType,
    SupportStatus,
)
from factuality_harness.domain.verdicts import ClaimVerdict, FinalAnswer, Verdict
from factuality_harness.evals.cases import EvalCase, ExpectedOutcome
from factuality_harness.evals.scoring import aggregate, score_case


def _final_with(answer: str, table: EvidenceTable) -> FinalAnswer:
    return FinalAnswer(
        answer=answer,
        confidence_summary="x",
        unsupported_or_uncertain_claims=[],
        evidence_table=table,
        audit_id="audit_t",
    )


def _build_trace(
    *,
    claim_text: str,
    claim_type: EpistemicType,
    verdict: Verdict,
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
    evidence_source: SourceType | None = None,
    contradictions: list[str] | None = None,
):
    claim = Claim(
        text=claim_text,
        parent_question=claim_text,
        epistemic_type=claim_type,
    )
    evidence = []
    if evidence_source is not None:
        evidence.append(
            Evidence(
                claim_id=claim.id,
                source_type=evidence_source,
                source_name="t",
                quote_or_result="x",
                supports_claim=SupportStatus.SUPPORTS,
                source_quality=SourceQuality.AUTHORITATIVE,
            )
        )
    cv = ClaimVerdict(
        claim_id=claim.id,
        verdict=verdict,
        confidence=confidence,
        rationale="r",
    )
    table = EvidenceTable(
        original_question=claim_text,
        claims=[claim],
        evidence=evidence,
        verdicts=[cv],
    )
    trace = AuditTrace(
        original_question=claim_text,
        decomposed_claims=[claim],
        retrieved_evidence=evidence,
        verdicts=[cv],
        contradictions_found=contradictions or [],
    )
    return claim, trace, table


def test_case_passes_when_every_check_holds():
    case = EvalCase(
        name="ok",
        question="What is the percentage increase from 100 to 125?",
        expected=ExpectedOutcome(
            first_claim_type="NUMERICAL",
            verdicts_must_include=["COMPUTED"],
            verdicts_must_not_include=["UNSUPPORTED"],
            answer_must_contain=["25"],
            evidence_source_types_must_include=["COMPUTATION"],
        ),
    )
    _, trace, table = _build_trace(
        claim_text="x",
        claim_type=EpistemicType.NUMERICAL,
        verdict=Verdict.COMPUTED,
        evidence_source=SourceType.COMPUTATION,
    )
    final = _final_with("the answer is 25%", table)

    result = score_case(case, final, trace)
    assert result.passed is True
    assert result.evaluated_count == 5
    assert all(c.passed for c in result.checks)


def test_case_fails_on_forbidden_verdict():
    case = EvalCase(
        name="overclaim",
        question="x",
        expected=ExpectedOutcome(
            verdicts_must_not_include=["VERIFIED"],
        ),
    )
    _, trace, table = _build_trace(
        claim_text="x",
        claim_type=EpistemicType.PREDICTIVE,
        verdict=Verdict.VERIFIED,  # this is the overclaim
    )
    result = score_case(case, _final_with("text", table), trace)
    assert result.passed is False
    assert any("verdict_excludes" in c.name for c in result.failed_checks)


def test_max_confidence_enforced():
    case = EvalCase(
        name="conf",
        question="x",
        expected=ExpectedOutcome(max_confidence="LOW"),
    )
    _, trace, table = _build_trace(
        claim_text="x",
        claim_type=EpistemicType.PREDICTIVE,
        verdict=Verdict.SUPPORTED,
        confidence=ConfidenceLevel.HIGH,  # exceeds the LOW ceiling
    )
    result = score_case(case, _final_with("t", table), trace)
    assert result.passed is False


def test_contradiction_expectation():
    case = EvalCase(
        name="contradict",
        question="x",
        expected=ExpectedOutcome(contradiction_expected=True),
    )
    _, trace_with, table = _build_trace(
        claim_text="x",
        claim_type=EpistemicType.PROCEDURAL,
        verdict=Verdict.CONTRADICTED,
        contradictions=["disagree"],
    )
    assert score_case(case, _final_with("x", table), trace_with).passed is True

    _, trace_without, table = _build_trace(
        claim_text="x",
        claim_type=EpistemicType.PROCEDURAL,
        verdict=Verdict.SUPPORTED,
        contradictions=[],
    )
    assert score_case(case, _final_with("x", table), trace_without).passed is False


def test_aggregate_metrics_count_correctly():
    cases = [
        # case 1 — passes everything
        EvalCase(
            name="ok",
            question="x",
            expected=ExpectedOutcome(
                first_claim_type="NUMERICAL",
                verdicts_must_include=["COMPUTED"],
            ),
        ),
        # case 2 — overclaim (forbidden verdict appears)
        EvalCase(
            name="overclaim",
            question="x",
            expected=ExpectedOutcome(
                first_claim_type="PREDICTIVE",
                verdicts_must_not_include=["VERIFIED"],
            ),
        ),
    ]

    _, trace_ok, table_ok = _build_trace(
        claim_text="x",
        claim_type=EpistemicType.NUMERICAL,
        verdict=Verdict.COMPUTED,
    )
    _, trace_bad, table_bad = _build_trace(
        claim_text="x",
        claim_type=EpistemicType.PREDICTIVE,
        verdict=Verdict.VERIFIED,
    )
    results = [
        score_case(cases[0], _final_with("ok", table_ok), trace_ok),
        score_case(cases[1], _final_with("ok", table_bad), trace_bad),
    ]
    summary = aggregate(results)
    assert summary.total_cases == 2
    assert summary.passed_cases == 1
    assert summary.failed_cases == 1
    assert summary.metrics["case_pass_rate"] == 0.5
    assert summary.metrics["overclaim_rate"] == 0.5
    # Both cases asserted on first_claim_type and both got it right.
    assert summary.metrics["classification_accuracy"] == 1.0
