"""Forecast tool backed by ``statsforecast``'s AutoARIMA.

Produces calibrated forecast evidence for PREDICTIVE claims. The verdict
calibrator never marks a predictive claim ``VERIFIED`` regardless of what
this tool emits — that's the spec. What this tool *does* provide is a
real point forecast plus a confidence interval, so the harness's draft
can report uncertainty honestly instead of refusing every prediction with
``UNCLEAR``.

Expected context shape::

    request.context["forecast"] = {
        "series": [{"timestamp": "2024-01-01", "value": 100.0}, ...],
        "horizon": 8,                  # periods to forecast
        "frequency": "W",              # pandas freq alias (D, W, M, Q, Y, ...)
        "confidence_level": 95,        # 80, 90, 95, 99 (default 95)
        # Optional — used to score the forecast against the claim:
        "baseline": 120.0,             # most recent comparison value
        "direction": "up",             # "up" / "down" — what the claim asks
    }

If ``baseline`` and ``direction`` are present, the tool sets
``supports_claim`` based on whether the prediction interval clearly clears
the baseline in the requested direction. Otherwise it returns
``PARTIALLY_SUPPORTS`` with the raw forecast attached.
"""

from __future__ import annotations

import warnings
from typing import Any

from ...domain.evidence import (
    FreshnessStatus,
    SourceQuality,
    SourceType,
    SupportStatus,
)
from .base import Tool, ToolRequest, ToolResult, build_evidence


_MIN_SERIES_LEN = 10


def _direction_supports(
    direction: str | None,
    baseline: float | None,
    forecast_lower: float,
    forecast_upper: float,
) -> SupportStatus:
    if direction is None or baseline is None:
        return SupportStatus.PARTIALLY_SUPPORTS
    direction = direction.lower().strip()
    if direction == "up":
        if forecast_lower > baseline:
            return SupportStatus.SUPPORTS
        if forecast_upper < baseline:
            return SupportStatus.CONTRADICTS
        return SupportStatus.PARTIALLY_SUPPORTS
    if direction == "down":
        if forecast_upper < baseline:
            return SupportStatus.SUPPORTS
        if forecast_lower > baseline:
            return SupportStatus.CONTRADICTS
        return SupportStatus.PARTIALLY_SUPPORTS
    return SupportStatus.PARTIALLY_SUPPORTS


class ForecastTool:
    name = "forecast"

    def __init__(
        self,
        *,
        default_horizon: int = 8,
        default_confidence_level: int = 95,
        default_frequency: str = "D",
        n_jobs: int = 1,
    ) -> None:
        self._horizon = default_horizon
        self._confidence_level = default_confidence_level
        self._frequency = default_frequency
        self._n_jobs = n_jobs

    def run(self, request: ToolRequest) -> ToolResult:
        spec = request.context.get("forecast")
        if not isinstance(spec, dict):
            return ToolResult(
                succeeded=False,
                error="ForecastTool requires request.context['forecast'].",
            )

        series = spec.get("series")
        if not isinstance(series, list) or len(series) < _MIN_SERIES_LEN:
            return ToolResult(
                succeeded=False,
                error=(
                    f"ForecastTool requires a series with at least {_MIN_SERIES_LEN} "
                    "observations as context['forecast']['series']."
                ),
            )

        horizon = int(spec.get("horizon", self._horizon))
        frequency = str(spec.get("frequency", self._frequency))
        confidence_level = int(spec.get("confidence_level", self._confidence_level))
        baseline = spec.get("baseline")
        direction = spec.get("direction")

        # Lazy imports keep the package import-time light.
        try:
            import pandas as pd
            from statsforecast import StatsForecast
            from statsforecast.models import AutoARIMA
        except ImportError as e:
            return ToolResult(
                succeeded=False,
                error=f"ForecastTool dependencies missing: {e}",
            )

        try:
            df = pd.DataFrame(
                {
                    "unique_id": "series",
                    "ds": [pd.to_datetime(p["timestamp"]) for p in series],
                    "y": [float(p["value"]) for p in series],
                }
            )
        except (KeyError, ValueError, TypeError) as e:
            return ToolResult(
                succeeded=False,
                error=f"ForecastTool: malformed series points ({e}).",
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sf = StatsForecast(models=[AutoARIMA()], freq=frequency, n_jobs=self._n_jobs)
                fcst = sf.forecast(df=df, h=horizon, level=[confidence_level])
        except Exception as e:  # statsforecast raises a variety of types
            return ToolResult(succeeded=False, error=f"Forecast failed: {e}")

        mean_col = "AutoARIMA"
        lo_col = f"AutoARIMA-lo-{confidence_level}"
        hi_col = f"AutoARIMA-hi-{confidence_level}"
        if mean_col not in fcst.columns:
            return ToolResult(
                succeeded=False,
                error="Forecast model produced no usable output.",
            )

        means = [float(v) for v in fcst[mean_col].tolist()]
        lows = [float(v) for v in fcst[lo_col].tolist()]
        highs = [float(v) for v in fcst[hi_col].tolist()]
        timestamps = [str(t) for t in fcst["ds"].tolist()]

        # Aggregate for the support/contradict test: use the average forecast
        # over the horizon and the interval's outer envelope.
        avg_mean = sum(means) / len(means)
        envelope_lo = min(lows)
        envelope_hi = max(highs)

        baseline_val: float | None = None
        if baseline is not None:
            try:
                baseline_val = float(baseline)
            except (ValueError, TypeError):
                baseline_val = None

        support = _direction_supports(
            direction if isinstance(direction, str) else None,
            baseline_val,
            envelope_lo,
            envelope_hi,
        )

        summary_parts = [
            f"AutoARIMA forecast over {horizon} period(s) at frequency {frequency!r}",
            f"point mean = {avg_mean:.4f}",
            f"{confidence_level}% CI envelope = [{envelope_lo:.4f}, {envelope_hi:.4f}]",
        ]
        if baseline_val is not None:
            summary_parts.append(f"baseline = {baseline_val:.4f}")
            if isinstance(direction, str):
                summary_parts.append(f"direction = {direction.lower()}")

        return ToolResult(
            evidence=[
                build_evidence(
                    claim_id=request.claim.id,
                    source_type=SourceType.FORECAST_MODEL,
                    source_name="auto_arima",
                    quote_or_result="; ".join(summary_parts),
                    normalized_result={
                        "model": "AutoARIMA",
                        "frequency": frequency,
                        "horizon": horizon,
                        "confidence_level": confidence_level,
                        "timestamps": timestamps,
                        "mean": means,
                        "lower": lows,
                        "upper": highs,
                        "summary": {
                            "avg_mean": avg_mean,
                            "envelope_lower": envelope_lo,
                            "envelope_upper": envelope_hi,
                        },
                        "baseline": baseline_val,
                        "direction": direction if isinstance(direction, str) else None,
                    },
                    supports_claim=support,
                    source_quality=SourceQuality.SECONDARY,
                    freshness=FreshnessStatus.NOT_TIME_SENSITIVE,
                    notes=(
                        "Forecast intervals reflect model uncertainty only; "
                        "they do not capture regime change or specification error."
                    ),
                )
            ]
        )


_: Tool = ForecastTool()
