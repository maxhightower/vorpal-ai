"""Per-case scoring + aggregate metrics.

Each ``ExpectedOutcome`` field that a case actually sets becomes a
``CheckResult`` after running. Aggregate metrics roll up across cases.

Why not boolean per-case: a case with 5 assertions that fails 1 is a
different signal than a case that fails all 5. We track per-check pass
rates so regressions point at the specific assertion type that broke.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..domain.audit import AuditTrace
from ..domain.confidence import ConfidenceLevel
from ..domain.verdicts import FinalAnswer
from .cases import EvalCase, ExpectedOutcome


_CONFIDENCE_RANK = {
    ConfidenceLevel.UNKNOWN: 0,
    ConfidenceLevel.LOW: 1,
    ConfidenceLevel.MEDIUM: 2,
    ConfidenceLevel.HIGH: 3,
}


# ---------------------------------------------------------------------------
# Per-check / per-case structures
# ---------------------------------------------------------------------------


class CheckResult(BaseModel):
    name: str
    passed: bool
    expected: Any | None = None
    actual: Any | None = None
    detail: str | None = None


class CaseResult(BaseModel):
    name: str
    passed: bool  # True iff every evaluated check passed
    checks: list[CheckResult] = Field(default_factory=list)
    error: str | None = None  # set when the harness raised before producing a verdict

    @property
    def evaluated_count(self) -> int:
        return len(self.checks)

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


class EvalSummary(BaseModel):
    total_cases: int
    passed_cases: int
    failed_cases: int
    case_pass_rate: float
    metrics: dict[str, float] = Field(default_factory=dict)
    cases: list[CaseResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-case scoring
# ---------------------------------------------------------------------------


def _confidence_at_most(actual: str, ceiling: str) -> bool:
    try:
        return _CONFIDENCE_RANK[ConfidenceLevel(actual)] <= _CONFIDENCE_RANK[
            ConfidenceLevel(ceiling)
        ]
    except KeyError:
        return False


def score_case(
    case: EvalCase, final: FinalAnswer | None, trace: AuditTrace | None, error: str | None = None
) -> CaseResult:
    if final is None or trace is None:
        return CaseResult(
            name=case.name,
            passed=False,
            checks=[],
            error=error or "harness produced no result",
        )

    expected = case.expected
    checks: list[CheckResult] = []

    claim_types = [
        c.epistemic_type.value for c in trace.decomposed_claims
    ]
    verdict_values = [v.verdict.value for v in trace.verdicts]
    confidence_values = [v.confidence.value for v in trace.verdicts]
    evidence_source_types = {
        e.source_type.value for e in trace.retrieved_evidence
    }
    answer_lower = (final.answer or "").lower()

    # Decomposition / classification
    if expected.claim_count is not None:
        ok = len(trace.decomposed_claims) == expected.claim_count
        checks.append(
            CheckResult(
                name="claim_count",
                passed=ok,
                expected=expected.claim_count,
                actual=len(trace.decomposed_claims),
            )
        )

    if expected.first_claim_type is not None:
        actual = claim_types[0] if claim_types else None
        checks.append(
            CheckResult(
                name="first_claim_type",
                passed=actual == expected.first_claim_type,
                expected=expected.first_claim_type,
                actual=actual,
            )
        )

    if expected.claim_types_in_order is not None:
        ok = claim_types == expected.claim_types_in_order
        checks.append(
            CheckResult(
                name="claim_types_in_order",
                passed=ok,
                expected=expected.claim_types_in_order,
                actual=claim_types,
            )
        )

    # Verdicts
    for required in expected.verdicts_must_include:
        checks.append(
            CheckResult(
                name=f"verdict_includes:{required}",
                passed=required in verdict_values,
                expected=required,
                actual=verdict_values,
            )
        )
    for forbidden in expected.verdicts_must_not_include:
        checks.append(
            CheckResult(
                name=f"verdict_excludes:{forbidden}",
                passed=forbidden not in verdict_values,
                expected=f"NOT {forbidden}",
                actual=verdict_values,
            )
        )

    if expected.max_confidence is not None:
        ok = all(
            _confidence_at_most(c, expected.max_confidence) for c in confidence_values
        )
        checks.append(
            CheckResult(
                name=f"max_confidence:{expected.max_confidence}",
                passed=ok,
                expected=expected.max_confidence,
                actual=confidence_values,
            )
        )

    # Final answer
    for substr in expected.answer_must_contain:
        ok = substr.lower() in answer_lower
        checks.append(
            CheckResult(
                name=f"answer_contains:{substr}",
                passed=ok,
                expected=substr,
                actual=None if ok else "<not found>",
            )
        )
    for substr in expected.answer_must_not_contain:
        ok = substr.lower() not in answer_lower
        checks.append(
            CheckResult(
                name=f"answer_excludes:{substr}",
                passed=ok,
                expected=f"NOT {substr}",
                actual=None if ok else "<found>",
            )
        )

    # Contradictions
    if expected.contradiction_expected is not None:
        actual = bool(trace.contradictions_found)
        checks.append(
            CheckResult(
                name="contradiction_detected",
                passed=actual == expected.contradiction_expected,
                expected=expected.contradiction_expected,
                actual=actual,
            )
        )

    # Evidence
    for required in expected.evidence_source_types_must_include:
        ok = required in evidence_source_types
        checks.append(
            CheckResult(
                name=f"evidence_source_includes:{required}",
                passed=ok,
                expected=required,
                actual=sorted(evidence_source_types),
            )
        )

    return CaseResult(
        name=case.name,
        passed=all(c.passed for c in checks) if checks else True,
        checks=checks,
    )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator > 0 else 0.0


def aggregate(case_results: list[CaseResult]) -> EvalSummary:
    """Compute the headline metrics from per-case results.

    Metrics defined here (all in [0, 1]):
      - case_pass_rate            fraction of cases with no failed checks.
      - classification_accuracy   among cases that asserted on first_claim_type or
                                  claim_types_in_order, fraction that passed.
      - verdict_correctness_rate  among cases with verdict assertions, fraction that
                                  passed all of them.
      - overclaim_rate            cases that produced a forbidden verdict
                                  (e.g. VERIFIED on a PREDICTIVE claim).
                                  Lower is better.
      - hallucination_rate        cases where a forbidden answer substring appeared.
                                  Lower is better.
    """
    total = len(case_results)
    passed_cases = sum(1 for r in case_results if r.passed)

    classification_evaluated = 0
    classification_passed = 0

    verdict_evaluated = 0
    verdict_passed = 0

    overclaim_count = 0
    hallucination_count = 0

    for r in case_results:
        had_classification = False
        had_verdict = False
        passed_classification = True
        passed_verdict = True
        case_overclaim = False
        case_hallucination = False

        for c in r.checks:
            if c.name in ("first_claim_type", "claim_types_in_order"):
                had_classification = True
                passed_classification = passed_classification and c.passed
            if c.name.startswith("verdict_includes:") or c.name.startswith(
                "verdict_excludes:"
            ):
                had_verdict = True
                passed_verdict = passed_verdict and c.passed
                if not c.passed and c.name.startswith("verdict_excludes:"):
                    case_overclaim = True
            if c.name.startswith("answer_excludes:") and not c.passed:
                case_hallucination = True

        if had_classification:
            classification_evaluated += 1
            if passed_classification:
                classification_passed += 1
        if had_verdict:
            verdict_evaluated += 1
            if passed_verdict:
                verdict_passed += 1
        if case_overclaim:
            overclaim_count += 1
        if case_hallucination:
            hallucination_count += 1

    return EvalSummary(
        total_cases=total,
        passed_cases=passed_cases,
        failed_cases=total - passed_cases,
        case_pass_rate=_rate(passed_cases, total),
        metrics={
            "case_pass_rate": _rate(passed_cases, total),
            "classification_accuracy": _rate(classification_passed, classification_evaluated),
            "verdict_correctness_rate": _rate(verdict_passed, verdict_evaluated),
            "overclaim_rate": _rate(overclaim_count, total),
            "hallucination_rate": _rate(hallucination_count, total),
        },
        cases=case_results,
    )
