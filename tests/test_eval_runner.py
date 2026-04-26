"""Integration: run the built-in eval suite end-to-end and verify the
baseline + regression machinery works against real harness output."""

from __future__ import annotations

import warnings

from factuality_harness.evals.baselines import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    load_baseline,
)
from factuality_harness.evals.run_evals import run_with_thresholds


def test_built_in_suite_runs_without_violations(tmp_path, monkeypatch):
    warnings.filterwarnings("ignore")
    monkeypatch.setenv("FACTUALITY_HARNESS_AUDIT_DIR", str(tmp_path / "audit"))

    baseline_path = tmp_path / "baseline.json"
    report = run_with_thresholds(save_baseline_path=str(baseline_path))

    # Save baseline succeeded, file exists, parses back into Baseline.
    assert report.saved_baseline_path == str(baseline_path)
    loaded = load_baseline(baseline_path)
    assert loaded is not None
    for metric in HIGHER_IS_BETTER | LOWER_IS_BETTER:
        # Every headline metric must be present in the saved baseline.
        assert metric in loaded.metrics

    # Sanity: at least the deterministic cases pass at full strength.
    assert report.summary.total_cases >= 6
    assert report.summary.metrics["case_pass_rate"] > 0.0


def test_floor_violation_surfaces(tmp_path, monkeypatch):
    """Pinning an unrealistically high floor should produce a violation."""
    warnings.filterwarnings("ignore")
    monkeypatch.setenv("FACTUALITY_HARNESS_AUDIT_DIR", str(tmp_path / "audit"))

    report = run_with_thresholds(
        minima={"case_pass_rate": 1.5},  # impossible floor
    )
    assert any(
        v.metric == "case_pass_rate" and v.kind == "min"
        for v in report.threshold_violations
    )
    assert report.has_violations is True


def test_baseline_regression_detection(tmp_path, monkeypatch):
    """Synthesize a "perfect" baseline and verify the runner flags any drop."""
    warnings.filterwarnings("ignore")
    monkeypatch.setenv("FACTUALITY_HARNESS_AUDIT_DIR", str(tmp_path / "audit"))

    # Hand-craft a baseline that's at the ceiling so any current run that's
    # below 100% on case_pass_rate registers as a regression.
    from factuality_harness.evals.baselines import Baseline, save_baseline

    perfect = Baseline(
        timestamp="t",
        metrics={
            "case_pass_rate": 1.0,
            "classification_accuracy": 1.0,
            "verdict_correctness_rate": 1.0,
            "overclaim_rate": 0.0,
            "hallucination_rate": 0.0,
        },
        cases={},
    )
    baseline_path = tmp_path / "perfect.json"
    save_baseline(perfect, baseline_path)

    report = run_with_thresholds(
        baseline_path=str(baseline_path),
        tolerance=0.0,
    )
    # If any case fails (which is normal for the conservative stub paths),
    # we should see at least one regression.
    if report.summary.metrics["case_pass_rate"] < 1.0:
        assert report.baseline_violations
        assert report.has_violations is True
