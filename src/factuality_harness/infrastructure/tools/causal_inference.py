"""Causal-inference tool backed by DoWhy.

Extends the harness beyond randomized A/B tests (handled by
``ABTestCausalTool``) to observational and quasi-experimental data via
backdoor adjustment, instrumental variables, and (when available)
synthetic-control / DiD methods. The output shape is the same as the
A/B tool — ``CAUSAL_MODEL`` evidence — so the verdict calibrator's CAUSAL
branch handles either interchangeably.

Expected context shape::

    request.context["causal_inference"] = {
        "method": "backdoor",            # backdoor | iv  (default: backdoor)
        "data": [{"X": 0.1, "T": 1, "Y": 2.3}, ...],  # rows
        "treatment": "T",
        "outcome": "Y",
        "common_causes": ["X", "Z"],     # for backdoor
        # OR for IV:
        # "instruments": ["IV"],
        "confidence_level": 95,
        "refute": True,                  # run a sanity refutation
    }

If the resulting confidence interval excludes 0, the evidence is marked
``SUPPORTS``. Otherwise ``INSUFFICIENT``. The refutation result is
included in the evidence record so the draft can flag fragile estimates.
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


_DEFAULT_METHOD = "backdoor"


def _ci_excludes_zero(low: float, high: float) -> bool:
    return (low > 0 and high > 0) or (low < 0 and high < 0)


def _resolve_estimator(method: str) -> str:
    method = (method or _DEFAULT_METHOD).strip().lower()
    if method in ("backdoor", "backdoor.linear_regression", "regression"):
        return "backdoor.linear_regression"
    if method in ("iv", "instrumental_variable", "iv.linear_regression"):
        return "iv.linear_regression"
    if method == "frontdoor":
        return "frontdoor.two_stage_regression"
    return method  # let DoWhy fail with a useful error


class CausalInferenceTool:
    name = "causal_inference"

    def run(self, request: ToolRequest) -> ToolResult:
        spec = request.context.get("causal_inference")
        if not isinstance(spec, dict):
            return ToolResult(
                succeeded=False,
                error="CausalInferenceTool requires request.context['causal_inference'].",
            )

        rows = spec.get("data")
        if not isinstance(rows, list) or len(rows) < 30:
            return ToolResult(
                succeeded=False,
                error=(
                    "CausalInferenceTool needs at least 30 data rows in "
                    "context['causal_inference']['data']."
                ),
            )

        treatment = spec.get("treatment")
        outcome = spec.get("outcome")
        if not isinstance(treatment, str) or not isinstance(outcome, str):
            return ToolResult(
                succeeded=False,
                error="treatment and outcome (column names) are required.",
            )

        common_causes = spec.get("common_causes") or []
        instruments = spec.get("instruments") or []
        if not isinstance(common_causes, list) or not isinstance(instruments, list):
            return ToolResult(
                succeeded=False,
                error="common_causes / instruments must be lists of column names.",
            )

        method_name = _resolve_estimator(str(spec.get("method") or _DEFAULT_METHOD))
        confidence_level = int(spec.get("confidence_level", 95))
        do_refute = bool(spec.get("refute", True))

        try:
            import pandas as pd
            from dowhy import CausalModel
        except ImportError as e:
            return ToolResult(
                succeeded=False,
                error=f"CausalInferenceTool dependencies missing: {e}",
            )

        try:
            df = pd.DataFrame(rows)
        except (TypeError, ValueError) as e:
            return ToolResult(
                succeeded=False,
                error=f"CausalInferenceTool: malformed data rows ({e}).",
            )

        for col in [treatment, outcome] + list(common_causes) + list(instruments):
            if col not in df.columns:
                return ToolResult(
                    succeeded=False,
                    error=f"Column {col!r} not present in supplied data.",
                )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = CausalModel(
                    data=df,
                    treatment=treatment,
                    outcome=outcome,
                    common_causes=common_causes or None,
                    instruments=instruments or None,
                )
                identified = model.identify_effect(proceed_when_unidentifiable=True)
                estimate = model.estimate_effect(
                    identified,
                    method_name=method_name,
                    confidence_intervals=True,
                    test_significance=True,
                )
        except Exception as e:
            return ToolResult(
                succeeded=False, error=f"Causal estimation failed: {e}"
            )

        # CI extraction across DoWhy versions.
        ci_low: float | None = None
        ci_high: float | None = None
        try:
            ci = estimate.get_confidence_intervals()
            # ci is typically [[lo, hi]] for a scalar estimate
            ci_low = float(ci[0][0])
            ci_high = float(ci[0][1])
        except Exception:
            pass

        p_value: float | None = None
        try:
            p_obj = estimate.test_stat_significance()
            raw_p = p_obj.get("p_value") if isinstance(p_obj, dict) else None
            if raw_p is not None:
                # Sometimes scalar, sometimes a 1-element array.
                p_value = float(getattr(raw_p, "__getitem__", lambda i: raw_p)(0))
        except Exception:
            pass

        effect = float(estimate.value)

        if ci_low is not None and ci_high is not None and _ci_excludes_zero(ci_low, ci_high):
            support = SupportStatus.SUPPORTS
        elif p_value is not None and p_value < (1 - confidence_level / 100.0):
            support = SupportStatus.SUPPORTS
        else:
            support = SupportStatus.INSUFFICIENT

        refutation_summary: dict[str, Any] | None = None
        if do_refute:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    refutation = model.refute_estimate(
                        identified,
                        estimate,
                        method_name="random_common_cause",
                    )
                refutation_summary = {
                    "method": "random_common_cause",
                    "new_effect": float(refutation.new_effect),
                    "estimated_effect_diff": float(refutation.new_effect) - effect,
                }
            except Exception as e:  # don't fail the whole call on refutation issues
                refutation_summary = {"error": str(e)}

        # Quasi-experimental designs (e.g. IV with strong instruments) get a
        # PRIMARY quality label; pure backdoor on observational data is SECONDARY.
        source_quality = (
            SourceQuality.PRIMARY
            if instruments
            else SourceQuality.SECONDARY
        )

        summary_bits = [
            f"DoWhy {method_name} estimate = {effect:+.4f}",
        ]
        if ci_low is not None and ci_high is not None:
            summary_bits.append(
                f"{confidence_level}% CI = [{ci_low:+.4f}, {ci_high:+.4f}]"
            )
        if p_value is not None:
            summary_bits.append(f"p = {p_value:.4g}")
        if refutation_summary and "new_effect" in refutation_summary:
            summary_bits.append(
                f"refutation new_effect = {refutation_summary['new_effect']:+.4f}"
            )

        return ToolResult(
            evidence=[
                build_evidence(
                    claim_id=request.claim.id,
                    source_type=SourceType.CAUSAL_MODEL,
                    source_name=f"dowhy:{method_name}",
                    quote_or_result="; ".join(summary_bits),
                    normalized_result={
                        "method": method_name,
                        "treatment": treatment,
                        "outcome": outcome,
                        "common_causes": common_causes,
                        "instruments": instruments,
                        "estimate": effect,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_value": p_value,
                        "n_rows": int(len(df)),
                        "refutation": refutation_summary,
                        "ci_excludes_zero": (
                            _ci_excludes_zero(ci_low, ci_high)
                            if ci_low is not None and ci_high is not None
                            else None
                        ),
                    },
                    supports_claim=support,
                    source_quality=source_quality,
                    freshness=FreshnessStatus.NOT_TIME_SENSITIVE,
                    notes=(
                        "Causal interpretation valid only if the assumed "
                        "identification (backdoor / IV / etc.) holds. Confounding "
                        "by unobserved variables is not detectable from data alone."
                    ),
                )
            ],
            metadata={"effect": effect, "support": support.value},
        )


_: Tool = CausalInferenceTool()
