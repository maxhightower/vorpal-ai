"""Typed eval cases.

The benchmark cases used to be loose dicts; now they're typed Pydantic
models. Each case names the question, optional documents/extra_context,
and a set of *structured assertions* about the harness's output. The
runner scores per assertion, aggregates, and compares to a baseline.

Why typed: the runner can't tell which assertions failed for which
reason without structure, and silent regressions are exactly what the
eval harness is supposed to catch.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExpectedOutcome(BaseModel):
    """Per-case assertions. Every field is optional — only set the ones
    that matter for the case. The scorer marks a check as ``passed`` only
    when the assertion was actually evaluated AND held."""

    # Decomposition / classification
    claim_count: int | None = None
    claim_types_in_order: list[str] | None = None  # e.g. ["NUMERICAL", "CAUSAL"]
    first_claim_type: str | None = None

    # Verdicts
    verdicts_must_include: list[str] = Field(default_factory=list)
    verdicts_must_not_include: list[str] = Field(default_factory=list)
    max_confidence: str | None = None  # e.g. "LOW" for predictions

    # Final answer text
    answer_must_contain: list[str] = Field(default_factory=list)
    answer_must_not_contain: list[str] = Field(default_factory=list)

    # Contradiction detection
    contradiction_expected: bool | None = None

    # Evidence
    evidence_source_types_must_include: list[str] = Field(default_factory=list)


class EvalCase(BaseModel):
    name: str
    question: str
    documents: list[dict[str, Any]] = Field(default_factory=list)
    extra_context: dict[str, Any] = Field(default_factory=dict)
    expected: ExpectedOutcome


# ---------------------------------------------------------------------------
# Built-in benchmark cases
# ---------------------------------------------------------------------------


CASES: list[EvalCase] = [
    EvalCase(
        name="numerical_percentage_change",
        question="What is the percentage increase from 100 to 125?",
        expected=ExpectedOutcome(
            claim_count=1,
            first_claim_type="NUMERICAL",
            verdicts_must_include=["COMPUTED"],
            verdicts_must_not_include=["UNSUPPORTED", "UNCLEAR"],
            answer_must_contain=["25"],
            evidence_source_types_must_include=["COMPUTATION"],
        ),
    ),
    EvalCase(
        name="causal_without_evidence",
        question="Did the new ad campaign cause sales to increase?",
        expected=ExpectedOutcome(
            first_claim_type="CAUSAL",
            verdicts_must_include=["UNSUPPORTED"],
            verdicts_must_not_include=["VERIFIED", "SUPPORTED"],
            max_confidence="LOW",
        ),
    ),
    EvalCase(
        name="predictive_without_forecast_data",
        question="Will demand increase next quarter?",
        expected=ExpectedOutcome(
            first_claim_type="PREDICTIVE",
            verdicts_must_not_include=["VERIFIED"],
            max_confidence="LOW",
        ),
    ),
    EvalCase(
        name="procedural_without_policy",
        question="Is this action allowed under the policy?",
        expected=ExpectedOutcome(
            first_claim_type="PROCEDURAL",
            verdicts_must_include=["UNCLEAR"],
        ),
    ),
    EvalCase(
        name="contradiction_remote_work",
        question="Can employees work remotely full-time?",
        documents=[
            {
                "name": "DocA",
                "text": "Employees may work remotely up to five days per week.",
                "effective_date": "2024-01-01",
            },
            {
                "name": "DocB",
                "text": (
                    "As of March 2026, employees must work in office "
                    "three days per week."
                ),
                "effective_date": "2026-03-01",
            },
        ],
        expected=ExpectedOutcome(
            contradiction_expected=True,
            answer_must_contain=["office"],
        ),
    ),
    EvalCase(
        name="unsupported_synthesis_best_option",
        question="Summarize this product and say whether it is the best option.",
        expected=ExpectedOutcome(
            verdicts_must_include=["UNCLEAR"],
            verdicts_must_not_include=["VERIFIED"],
        ),
    ),
    # New: causal claim WITH experimental data should now pass to SUPPORTED.
    EvalCase(
        name="causal_with_ab_experiment",
        question="Did the Spring 2026 email campaign cause higher purchase conversion?",
        extra_context={
            "experiment": {
                "name": "spring_2026_email",
                "metric": "purchase_conversion_7d",
                "design": "randomized A/B test, 50/50",
                "alpha": 0.05,
                "treatment": {"n": 50_000, "conversions": 1_250},
                "control": {"n": 50_000, "conversions": 1_000},
            }
        },
        expected=ExpectedOutcome(
            first_claim_type="CAUSAL",
            verdicts_must_include=["SUPPORTED"],
            verdicts_must_not_include=["UNSUPPORTED"],
            evidence_source_types_must_include=["CAUSAL_MODEL"],
        ),
    ),
    # New: optimization claim with explicit objective + constraints.
    EvalCase(
        name="optimization_with_explicit_model",
        question="What is the optimal allocation between search and social spend?",
        extra_context={
            "optimization": {
                "kind": "linear",
                "objective": {"sense": "maximize", "coefficients": [0.04, 0.07]},
                "variables": [
                    {"name": "search", "lower": 0},
                    {"name": "social", "lower": 0},
                ],
                "constraints": [
                    {"coefficients": [1, 1], "sense": "<=", "rhs": 100_000},
                    {"coefficients": [1, 0], "sense": ">=", "rhs": 20_000},
                    {"coefficients": [0, 1], "sense": "<=", "rhs": 60_000},
                ],
            }
        },
        expected=ExpectedOutcome(
            first_claim_type="OPTIMIZATION",
            verdicts_must_include=["SUPPORTED"],
            evidence_source_types_must_include=["OPTIMIZER"],
        ),
    ),
]
