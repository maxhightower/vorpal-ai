from factuality_harness.domain.claims import Claim
from factuality_harness.domain.epistemic_types import EpistemicType
from factuality_harness.domain.evidence import SourceType, SupportStatus
from factuality_harness.infrastructure.tools.ab_test import ABTestCausalTool
from factuality_harness.infrastructure.tools.base import ToolRequest


def _causal_claim() -> Claim:
    return Claim(
        text="did the campaign cause the increase",
        parent_question="x",
        epistemic_type=EpistemicType.CAUSAL,
    )


def test_significant_lift_supports_claim():
    tool = ABTestCausalTool()
    req = ToolRequest(
        claim=_causal_claim(),
        context={
            "experiment": {
                "name": "demo",
                "metric": "conv",
                "treatment": {"n": 50_000, "conversions": 1_250},
                "control":   {"n": 50_000, "conversions": 1_000},
                "alpha": 0.05,
            }
        },
    )
    result = tool.run(req)
    assert result.succeeded
    ev = result.evidence[0]
    assert ev.source_type == SourceType.CAUSAL_MODEL
    assert ev.supports_claim == SupportStatus.SUPPORTS
    assert ev.normalized_result["significant"] is True
    assert ev.normalized_result["p_value"] < 0.05


def test_null_result_does_not_support():
    tool = ABTestCausalTool()
    # Tiny absolute lift, small sample: should not be significant.
    req = ToolRequest(
        claim=_causal_claim(),
        context={
            "experiment": {
                "name": "demo_null",
                "metric": "conv",
                "treatment": {"n": 1_000, "conversions": 102},
                "control":   {"n": 1_000, "conversions": 100},
                "alpha": 0.05,
            }
        },
    )
    result = tool.run(req)
    assert result.succeeded
    ev = result.evidence[0]
    assert ev.supports_claim == SupportStatus.INSUFFICIENT
    assert ev.normalized_result["significant"] is False


def test_missing_experiment_payload_fails_cleanly():
    tool = ABTestCausalTool()
    result = tool.run(ToolRequest(claim=_causal_claim()))
    assert not result.succeeded
    assert "experiment" in (result.error or "").lower()
