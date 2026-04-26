from __future__ import annotations

import pandas as pd
import pytest

from factuality_harness.domain.claims import Claim
from factuality_harness.domain.epistemic_types import EpistemicType
from factuality_harness.domain.evidence import SourceType, SupportStatus
from factuality_harness.infrastructure.tools.base import ToolRequest
from factuality_harness.infrastructure.tools.forecast import ForecastTool


def _claim() -> Claim:
    return Claim(
        text="Will the metric increase next quarter?",
        parent_question="x",
        epistemic_type=EpistemicType.PREDICTIVE,
    )


def _trending_series(slope: float = 0.5, n: int = 60, noise: float = 1.0):
    import numpy as np

    rng = np.random.RandomState(0)
    dates = pd.date_range("2024-01-01", periods=n, freq="W")
    y = 100 + slope * np.arange(n) + rng.normal(0, noise, n)
    return [{"timestamp": str(d.date()), "value": float(v)} for d, v in zip(dates, y)]


def test_forecasts_a_trending_series():
    tool = ForecastTool()
    req = ToolRequest(
        claim=_claim(),
        context={
            "forecast": {
                "series": _trending_series(slope=0.5),
                "horizon": 8,
                "frequency": "W",
                "confidence_level": 95,
            }
        },
    )
    result = tool.run(req)
    assert result.succeeded
    ev = result.evidence[0]
    assert ev.source_type == SourceType.FORECAST_MODEL
    nr = ev.normalized_result
    assert len(nr["mean"]) == 8
    assert len(nr["lower"]) == 8 and len(nr["upper"]) == 8
    # Linear trend extrapolation: forecast mean should be above the start.
    assert nr["summary"]["avg_mean"] > 100.0


def test_supports_when_lower_bound_exceeds_baseline():
    tool = ForecastTool()
    series = _trending_series(slope=1.0, n=80, noise=0.5)
    last_value = series[-1]["value"]
    req = ToolRequest(
        claim=_claim(),
        context={
            "forecast": {
                "series": series,
                "horizon": 8,
                "frequency": "W",
                "confidence_level": 80,  # tighter -> easier for envelope to clear baseline
                "baseline": last_value,
                "direction": "up",
            }
        },
    )
    result = tool.run(req)
    assert result.succeeded
    assert result.evidence[0].supports_claim == SupportStatus.SUPPORTS


def test_contradicts_when_direction_mismatched():
    tool = ForecastTool()
    series = _trending_series(slope=1.0, n=80, noise=0.5)
    last_value = series[-1]["value"]
    req = ToolRequest(
        claim=_claim(),
        context={
            "forecast": {
                "series": series,
                "horizon": 6,
                "frequency": "W",
                "confidence_level": 80,
                "baseline": last_value,
                "direction": "down",  # series is going up, so a down claim should contradict
            }
        },
    )
    result = tool.run(req)
    assert result.succeeded
    assert result.evidence[0].supports_claim == SupportStatus.CONTRADICTS


def test_too_short_series_fails():
    tool = ForecastTool()
    req = ToolRequest(
        claim=_claim(),
        context={
            "forecast": {
                "series": [{"timestamp": "2024-01-01", "value": 1.0}],
                "horizon": 4,
            }
        },
    )
    result = tool.run(req)
    assert not result.succeeded
    assert "10 observations" in (result.error or "")


def test_missing_context_fails_cleanly():
    tool = ForecastTool()
    result = tool.run(ToolRequest(claim=_claim()))
    assert not result.succeeded
    assert "context['forecast']" in (result.error or "")
