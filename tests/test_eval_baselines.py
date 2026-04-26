"""Baseline persistence + threshold/regression checks."""

from __future__ import annotations

from factuality_harness.evals.baselines import (
    Baseline,
    check_absolute_thresholds,
    check_regressions,
    load_baseline,
    make_baseline,
    save_baseline,
)
from factuality_harness.evals.scoring import CaseResult


def test_baseline_roundtrip(tmp_path):
    cases = [
        CaseResult(name="a", passed=True, checks=[]),
        CaseResult(name="b", passed=False, checks=[]),
    ]
    bl = make_baseline({"case_pass_rate": 0.5, "overclaim_rate": 0.0}, cases)
    path = tmp_path / "baseline.json"
    save_baseline(bl, path)

    loaded = load_baseline(path)
    assert isinstance(loaded, Baseline)
    assert loaded.metrics["case_pass_rate"] == 0.5
    assert loaded.cases == {"a": True, "b": False}


def test_load_baseline_returns_none_when_missing(tmp_path):
    assert load_baseline(tmp_path / "nope.json") is None


def test_absolute_floor_violation():
    metrics = {"classification_accuracy": 0.7}
    violations = check_absolute_thresholds(metrics, minima={"classification_accuracy": 0.8})
    assert len(violations) == 1
    assert violations[0].kind == "min"
    assert violations[0].metric == "classification_accuracy"


def test_absolute_floor_pass():
    violations = check_absolute_thresholds(
        {"classification_accuracy": 0.9}, minima={"classification_accuracy": 0.8}
    )
    assert violations == []


def test_absolute_ceiling_violation():
    violations = check_absolute_thresholds(
        {"overclaim_rate": 0.2}, maxima={"overclaim_rate": 0.1}
    )
    assert len(violations) == 1
    assert violations[0].kind == "max"


def test_regression_higher_is_better_drop_flagged():
    baseline = Baseline(
        timestamp="t",
        metrics={"classification_accuracy": 0.95},
        cases={},
    )
    current = {"classification_accuracy": 0.85}
    violations = check_regressions(current, baseline, tolerance=0.05)
    assert len(violations) == 1
    assert violations[0].kind == "regression"
    assert violations[0].metric == "classification_accuracy"


def test_regression_higher_is_better_within_tolerance():
    baseline = Baseline(
        timestamp="t", metrics={"classification_accuracy": 0.95}, cases={}
    )
    current = {"classification_accuracy": 0.91}  # 4pp drop
    violations = check_regressions(current, baseline, tolerance=0.05)
    assert violations == []


def test_regression_lower_is_better_rise_flagged():
    baseline = Baseline(timestamp="t", metrics={"overclaim_rate": 0.05}, cases={})
    current = {"overclaim_rate": 0.20}
    violations = check_regressions(current, baseline, tolerance=0.05)
    assert len(violations) == 1
    assert violations[0].metric == "overclaim_rate"


def test_regression_unknown_metric_ignored():
    baseline = Baseline(timestamp="t", metrics={"unknown": 0.5}, cases={})
    current = {"unknown": 0.0}
    # Not in HIGHER_IS_BETTER or LOWER_IS_BETTER -> not flagged.
    assert check_regressions(current, baseline, tolerance=0.0) == []
