"""Eval runner with baseline + threshold support.

Top-level API: ``run_all`` runs every built-in case through a freshly built
pipeline and returns a JSON-serializable dict (back-compat with the
existing CLI invocation). ``run_with_thresholds`` is the richer entry point
used by the CLI for regression checking.

The runner is deliberately tolerant of harness exceptions: a case that
crashes the harness is recorded as a failed case, not as an unhandled
exception. That keeps the eval CLI usable as a CI gate.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..application.pipeline import PipelineRequest
from ..infrastructure.retrieval.base import Document
from ..interfaces.factory import build_pipeline
from .baselines import (
    Baseline,
    ThresholdViolation,
    check_absolute_thresholds,
    check_regressions,
    make_baseline,
)
from .cases import CASES, EvalCase
from .scoring import CaseResult, EvalSummary, aggregate, score_case


def _run_one(pipeline, case: EvalCase) -> CaseResult:
    try:
        documents = [Document.model_validate(d) for d in case.documents]
        final = pipeline.run(
            PipelineRequest(
                question=case.question,
                documents=documents,
                extra_context=case.extra_context,
            )
        )
        trace = pipeline.audit_repo.get(final.audit_id)
        return score_case(case, final, trace)
    except Exception as e:  # never let one case crash the suite
        return score_case(case, None, None, error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def run_eval_cases(cases: list[EvalCase] | None = None) -> EvalSummary:
    pipeline = build_pipeline()
    use_cases = cases if cases is not None else CASES
    results = [_run_one(pipeline, c) for c in use_cases]
    return aggregate(results)


# Back-compat: existing CLI calls expected ``run_all() -> dict``.
def run_all() -> dict[str, Any]:
    return run_eval_cases().model_dump()


class EvalRunReport(BaseModel):
    summary: EvalSummary
    baseline_violations: list[ThresholdViolation] = Field(default_factory=list)
    threshold_violations: list[ThresholdViolation] = Field(default_factory=list)
    saved_baseline_path: str | None = None

    @property
    def has_violations(self) -> bool:
        return bool(self.baseline_violations or self.threshold_violations)


def run_with_thresholds(
    *,
    baseline_path: str | None = None,
    save_baseline_path: str | None = None,
    tolerance: float = 0.0,
    minima: dict[str, float] | None = None,
    maxima: dict[str, float] | None = None,
) -> EvalRunReport:
    """Run the suite, score, optionally save/compare to a baseline, and
    return a structured report. Caller decides whether violations should
    map to a non-zero exit code (the CLI does)."""
    summary = run_eval_cases()

    threshold_violations = check_absolute_thresholds(
        summary.metrics, minima=minima, maxima=maxima
    )

    baseline_violations: list[ThresholdViolation] = []
    if baseline_path:
        from .baselines import load_baseline

        existing = load_baseline(baseline_path)
        if existing is not None:
            baseline_violations = check_regressions(
                summary.metrics, existing, tolerance=tolerance
            )

    saved_path: str | None = None
    if save_baseline_path:
        new_baseline = make_baseline(summary.metrics, summary.cases)
        from .baselines import save_baseline

        save_baseline(new_baseline, save_baseline_path)
        saved_path = str(save_baseline_path)

    return EvalRunReport(
        summary=summary,
        baseline_violations=baseline_violations,
        threshold_violations=threshold_violations,
        saved_baseline_path=saved_path,
    )
