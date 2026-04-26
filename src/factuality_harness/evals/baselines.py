"""Baseline persistence and regression comparison for the eval harness.

A baseline is a snapshot of metric values from a prior run. The runner can
either save the current run as a new baseline, or compare against an
existing one and fail when any metric regresses beyond a configurable
tolerance.

Two flavors of threshold:

 - **Absolute floor / ceiling.** ``--min classification_accuracy=0.8``
   fails the run if the metric drops below 0.8. ``--max overclaim_rate=0.1``
   fails if it exceeds 0.1.

 - **Regression.** ``--baseline f.json --tolerance 0.05`` fails if any
   "higher-is-better" metric drops by more than 5 percentage points, or any
   "lower-is-better" metric rises by more than 5 percentage points, vs the
   stored baseline.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


# Metrics where higher is better (a drop is a regression).
HIGHER_IS_BETTER = {
    "case_pass_rate",
    "classification_accuracy",
    "verdict_correctness_rate",
}

# Metrics where lower is better (a rise is a regression).
LOWER_IS_BETTER = {
    "overclaim_rate",
    "hallucination_rate",
}


class Baseline(BaseModel):
    timestamp: str
    metrics: dict[str, float] = Field(default_factory=dict)
    cases: dict[str, bool] = Field(default_factory=dict)  # name -> passed


class ThresholdViolation(BaseModel):
    metric: str
    kind: str  # "min" | "max" | "regression"
    threshold: float
    actual: float
    detail: str


def make_baseline(metrics: dict[str, float], case_results: list) -> Baseline:
    return Baseline(
        timestamp=datetime.now(timezone.utc).isoformat(),
        metrics=dict(metrics),
        cases={r.name: r.passed for r in case_results},
    )


def save_baseline(baseline: Baseline, path: str | Path) -> None:
    Path(path).write_text(baseline.model_dump_json(indent=2), encoding="utf-8")


def load_baseline(path: str | Path) -> Baseline | None:
    p = Path(path)
    if not p.exists():
        return None
    return Baseline.model_validate_json(p.read_text(encoding="utf-8"))


def check_absolute_thresholds(
    metrics: dict[str, float],
    *,
    minima: dict[str, float] | None = None,
    maxima: dict[str, float] | None = None,
) -> list[ThresholdViolation]:
    out: list[ThresholdViolation] = []
    for name, floor in (minima or {}).items():
        actual = metrics.get(name)
        if actual is None:
            continue
        if actual < floor:
            out.append(
                ThresholdViolation(
                    metric=name,
                    kind="min",
                    threshold=floor,
                    actual=actual,
                    detail=f"{name}={actual:.4f} fell below floor {floor:.4f}.",
                )
            )
    for name, ceiling in (maxima or {}).items():
        actual = metrics.get(name)
        if actual is None:
            continue
        if actual > ceiling:
            out.append(
                ThresholdViolation(
                    metric=name,
                    kind="max",
                    threshold=ceiling,
                    actual=actual,
                    detail=f"{name}={actual:.4f} exceeded ceiling {ceiling:.4f}.",
                )
            )
    return out


def check_regressions(
    current: dict[str, float],
    baseline: Baseline,
    *,
    tolerance: float = 0.0,
) -> list[ThresholdViolation]:
    """Detect regressions vs. a stored baseline.

    A "regression" means the metric moved in the *worse* direction by more
    than ``tolerance`` percentage points (since metrics are in [0, 1], the
    tolerance is absolute on the same scale).
    """
    violations: list[ThresholdViolation] = []
    for name, baseline_val in baseline.metrics.items():
        current_val = current.get(name)
        if current_val is None:
            continue
        delta = current_val - baseline_val
        if name in HIGHER_IS_BETTER and delta < -tolerance:
            violations.append(
                ThresholdViolation(
                    metric=name,
                    kind="regression",
                    threshold=tolerance,
                    actual=delta,
                    detail=(
                        f"{name} dropped from {baseline_val:.4f} to "
                        f"{current_val:.4f} (Δ={delta:+.4f}, tolerance={tolerance:.4f})."
                    ),
                )
            )
        elif name in LOWER_IS_BETTER and delta > tolerance:
            violations.append(
                ThresholdViolation(
                    metric=name,
                    kind="regression",
                    threshold=tolerance,
                    actual=delta,
                    detail=(
                        f"{name} rose from {baseline_val:.4f} to "
                        f"{current_val:.4f} (Δ={delta:+.4f}, tolerance={tolerance:.4f})."
                    ),
                )
            )
    return violations
