"""Real A/B-test causal-evidence tool.

Consumes randomized-experiment data from the request context and emits a
``CAUSAL_MODEL`` evidence record whose ``supports_claim`` reflects the result
of a two-proportion z-test. Unlike ``CausalModelStub``, this tool produces
genuine causal evidence when given a properly randomized comparison.

Expected context shape::

    request.context["experiment"] = {
        "name": str,
        "metric": str,            # e.g. "purchase_conversion"
        "treatment": {"n": int, "conversions": int},
        "control":   {"n": int, "conversions": int},
        "design":    str,         # e.g. "randomized A/B test, 50/50"
        "start":     "YYYY-MM-DD",
        "end":       "YYYY-MM-DD",
        "alpha":     float,       # significance level, default 0.05
    }

Calculations (no scipy dependency):

    p1 = c1 / n1                       # treatment rate
    p2 = c2 / n2                       # control rate
    p_pool = (c1 + c2) / (n1 + n2)
    SE = sqrt( p_pool * (1 - p_pool) * (1/n1 + 1/n2) )
    z  = (p1 - p2) / SE
    p_value = 2 * (1 - Phi(|z|))
    95% CI for (p1 - p2) = (p1 - p2) +/- 1.96 * SE_diff
        SE_diff = sqrt( p1(1-p1)/n1 + p2(1-p2)/n2 )
"""

from __future__ import annotations

import math
from typing import Any

from ...domain.evidence import (
    FreshnessStatus,
    SourceQuality,
    SourceType,
    SupportStatus,
)
from .base import Tool, ToolRequest, ToolResult, build_evidence


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _two_proportion_z(c1: int, n1: int, c2: int, n2: int) -> dict[str, float]:
    p1 = c1 / n1
    p2 = c2 / n2
    p_pool = (c1 + c2) / (n1 + n2)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se_pool if se_pool > 0 else 0.0
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
    se_diff = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    ci_low = diff - 1.96 * se_diff
    ci_high = diff + 1.96 * se_diff
    return {
        "p1": p1,
        "p2": p2,
        "p_pool": p_pool,
        "se_pool": se_pool,
        "se_diff": se_diff,
        "z": z,
        "p_value": p_value,
        "abs_diff": diff,
        "rel_lift": (diff / p2) if p2 > 0 else float("nan"),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


class ABTestCausalTool:
    name = "ab_test"

    def run(self, request: ToolRequest) -> ToolResult:
        exp: dict[str, Any] | None = request.context.get("experiment")
        if not exp:
            return ToolResult(
                succeeded=False,
                error=(
                    "ABTestCausalTool requires request.context['experiment']. "
                    "No experimental data supplied for this causal claim."
                ),
            )

        try:
            t = exp["treatment"]
            c = exp["control"]
            stats = _two_proportion_z(
                int(t["conversions"]), int(t["n"]),
                int(c["conversions"]), int(c["n"]),
            )
        except (KeyError, ZeroDivisionError, ValueError) as e:
            return ToolResult(succeeded=False, error=f"Bad experiment payload: {e}")

        alpha = float(exp.get("alpha", 0.05))
        significant = stats["p_value"] < alpha

        support = SupportStatus.SUPPORTS if significant else SupportStatus.INSUFFICIENT

        summary = (
            f"Randomized {exp.get('design', 'A/B test')} on metric "
            f"{exp.get('metric', '<metric>')!r}: "
            f"treatment {stats['p1']*100:.3f}% (n={t['n']}) vs control "
            f"{stats['p2']*100:.3f}% (n={c['n']}); "
            f"absolute diff = {stats['abs_diff']*100:+.3f} pp, "
            f"relative lift = {stats['rel_lift']*100:+.2f}%, "
            f"z = {stats['z']:.3f}, p = {stats['p_value']:.4g}, "
            f"95% CI for diff = [{stats['ci_low']*100:+.3f} pp, "
            f"{stats['ci_high']*100:+.3f} pp], alpha = {alpha}, "
            f"significant = {significant}."
        )

        return ToolResult(
            evidence=[
                build_evidence(
                    claim_id=request.claim.id,
                    source_type=SourceType.CAUSAL_MODEL,
                    source_name=f"ab_test:{exp.get('name', 'unnamed')}",
                    quote_or_result=summary,
                    normalized_result={
                        **stats,
                        "alpha": alpha,
                        "significant": significant,
                        "design": exp.get("design"),
                        "start": exp.get("start"),
                        "end": exp.get("end"),
                        "metric": exp.get("metric"),
                        "n_treatment": t["n"],
                        "n_control": c["n"],
                        "conversions_treatment": t["conversions"],
                        "conversions_control": c["conversions"],
                    },
                    supports_claim=support,
                    source_quality=SourceQuality.PRIMARY,
                    freshness=FreshnessStatus.CURRENT,
                    notes=(
                        "Two-proportion z-test on randomized A/B comparison. "
                        "Causal interpretation valid if randomization integrity holds."
                    ),
                )
            ],
            metadata={"significant": significant, "p_value": stats["p_value"]},
        )


# Static type compatibility check.
_: Tool = ABTestCausalTool()
