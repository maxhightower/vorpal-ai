from __future__ import annotations

import numpy as np
import pytest

from factuality_harness.domain.claims import Claim
from factuality_harness.domain.epistemic_types import EpistemicType
from factuality_harness.domain.evidence import SourceQuality, SourceType, SupportStatus
from factuality_harness.infrastructure.tools.base import ToolRequest
from factuality_harness.infrastructure.tools.causal_inference import (
    CausalInferenceTool,
)


def _claim() -> Claim:
    return Claim(
        text="did treatment cause outcome?",
        parent_question="x",
        epistemic_type=EpistemicType.CAUSAL,
    )


def _confounded_data(true_effect: float, n: int = 1500, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, n)
    T = (0.5 * X + rng.normal(0, 1, n) > 0).astype(int)
    Y = true_effect * T + 1.5 * X + rng.normal(0, 1, n)
    return [{"X": float(x), "T": int(t), "Y": float(y)} for x, t, y in zip(X, T, Y)]


def test_recovers_true_effect_via_backdoor():
    tool = CausalInferenceTool()
    rows = _confounded_data(true_effect=2.0)
    result = tool.run(
        ToolRequest(
            claim=_claim(),
            context={
                "causal_inference": {
                    "method": "backdoor",
                    "data": rows,
                    "treatment": "T",
                    "outcome": "Y",
                    "common_causes": ["X"],
                    "refute": False,  # speed up the test
                }
            },
        )
    )
    assert result.succeeded
    ev = result.evidence[0]
    assert ev.source_type == SourceType.CAUSAL_MODEL
    assert ev.source_quality == SourceQuality.SECONDARY
    assert ev.supports_claim == SupportStatus.SUPPORTS
    nr = ev.normalized_result
    assert abs(nr["estimate"] - 2.0) < 0.2
    assert nr["ci_excludes_zero"] is True


def test_null_effect_yields_insufficient():
    tool = CausalInferenceTool()
    rows = _confounded_data(true_effect=0.0)
    result = tool.run(
        ToolRequest(
            claim=_claim(),
            context={
                "causal_inference": {
                    "method": "backdoor",
                    "data": rows,
                    "treatment": "T",
                    "outcome": "Y",
                    "common_causes": ["X"],
                    "refute": False,
                }
            },
        )
    )
    assert result.succeeded
    ev = result.evidence[0]
    # 95% CI should contain 0, so support is INSUFFICIENT.
    nr = ev.normalized_result
    assert abs(nr["estimate"]) < 0.3
    assert ev.supports_claim == SupportStatus.INSUFFICIENT


def test_missing_data_columns_fails_cleanly():
    tool = CausalInferenceTool()
    result = tool.run(
        ToolRequest(
            claim=_claim(),
            context={
                "causal_inference": {
                    "data": [{"X": 0.0, "T": 0, "Y": 1.0}] * 50,
                    "treatment": "T",
                    "outcome": "Y",
                    "common_causes": ["NotAColumn"],
                }
            },
        )
    )
    assert not result.succeeded
    assert "NotAColumn" in (result.error or "")


def test_too_few_rows_fails_cleanly():
    tool = CausalInferenceTool()
    result = tool.run(
        ToolRequest(
            claim=_claim(),
            context={
                "causal_inference": {
                    "data": [{"X": 0.0, "T": 0, "Y": 1.0}] * 10,
                    "treatment": "T",
                    "outcome": "Y",
                    "common_causes": ["X"],
                }
            },
        )
    )
    assert not result.succeeded
    assert "30" in (result.error or "")


def test_missing_context_fails_cleanly():
    tool = CausalInferenceTool()
    result = tool.run(ToolRequest(claim=_claim()))
    assert not result.succeeded
    assert "context['causal_inference']" in (result.error or "")
