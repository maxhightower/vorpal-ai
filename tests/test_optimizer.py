from __future__ import annotations

from factuality_harness.domain.claims import Claim
from factuality_harness.domain.epistemic_types import EpistemicType
from factuality_harness.domain.evidence import SourceType, SupportStatus
from factuality_harness.infrastructure.tools.base import ToolRequest
from factuality_harness.infrastructure.tools.optimizer import LinearOptimizerTool


def _claim() -> Claim:
    return Claim(
        text="What is the best allocation?",
        parent_question="x",
        epistemic_type=EpistemicType.OPTIMIZATION,
    )


def test_known_optimum_recovered():
    """maximize 3x + 5y s.t. x+2y<=8, 2x+y<=8, x,y>=0.

    Optimum at x=8/3, y=8/3, objective = 64/3 ≈ 21.333.
    """
    tool = LinearOptimizerTool()
    req = ToolRequest(
        claim=_claim(),
        context={
            "optimization": {
                "kind": "linear",
                "objective": {"sense": "maximize", "coefficients": [3, 5]},
                "variables": [
                    {"name": "x", "lower": 0},
                    {"name": "y", "lower": 0},
                ],
                "constraints": [
                    {"coefficients": [1, 2], "sense": "<=", "rhs": 8},
                    {"coefficients": [2, 1], "sense": "<=", "rhs": 8},
                ],
            }
        },
    )
    result = tool.run(req)
    assert result.succeeded
    ev = result.evidence[0]
    assert ev.source_type == SourceType.OPTIMIZER
    assert ev.supports_claim == SupportStatus.SUPPORTS
    nr = ev.normalized_result
    assert abs(nr["objective_value"] - 64 / 3) < 1e-6
    assert abs(nr["solution"]["x"] - 8 / 3) < 1e-6
    assert abs(nr["solution"]["y"] - 8 / 3) < 1e-6


def test_minimize_with_lower_bound_constraint():
    """minimize 3x + 5y s.t. x+y >= 4, x,y >= 0.

    Optimum at x=4, y=0, objective = 12.
    """
    tool = LinearOptimizerTool()
    req = ToolRequest(
        claim=_claim(),
        context={
            "optimization": {
                "objective": {"sense": "minimize", "coefficients": [3, 5]},
                "variables": [
                    {"name": "x", "lower": 0},
                    {"name": "y", "lower": 0},
                ],
                "constraints": [
                    {"coefficients": [1, 1], "sense": ">=", "rhs": 4},
                ],
            }
        },
    )
    result = tool.run(req)
    assert result.succeeded
    nr = result.evidence[0].normalized_result
    assert abs(nr["objective_value"] - 12.0) < 1e-6


def test_equality_constraint():
    """minimize x + y s.t. x + y == 5, x>=0, y>=0. Optimum: any feasible, obj = 5."""
    tool = LinearOptimizerTool()
    result = tool.run(
        ToolRequest(
            claim=_claim(),
            context={
                "optimization": {
                    "objective": {"sense": "minimize", "coefficients": [1, 1]},
                    "variables": [
                        {"name": "x", "lower": 0},
                        {"name": "y", "lower": 0},
                    ],
                    "constraints": [
                        {"coefficients": [1, 1], "sense": "==", "rhs": 5},
                    ],
                }
            },
        )
    )
    assert result.succeeded
    assert abs(result.evidence[0].normalized_result["objective_value"] - 5.0) < 1e-6


def test_infeasible_returns_contradicts():
    """Contradictory constraints: x+y<=1 and x+y>=10, x,y>=0 — no feasible point."""
    tool = LinearOptimizerTool()
    result = tool.run(
        ToolRequest(
            claim=_claim(),
            context={
                "optimization": {
                    "objective": {"sense": "minimize", "coefficients": [1, 1]},
                    "variables": [
                        {"name": "x", "lower": 0},
                        {"name": "y", "lower": 0},
                    ],
                    "constraints": [
                        {"coefficients": [1, 1], "sense": "<=", "rhs": 1},
                        {"coefficients": [1, 1], "sense": ">=", "rhs": 10},
                    ],
                }
            },
        )
    )
    assert not result.succeeded
    # Infeasibility actively refutes any "optimal allocation" claim.
    assert result.evidence[0].supports_claim == SupportStatus.CONTRADICTS


def test_unbounded_returns_contradicts():
    """minimize x with no lower bound -> unbounded."""
    tool = LinearOptimizerTool()
    result = tool.run(
        ToolRequest(
            claim=_claim(),
            context={
                "optimization": {
                    "objective": {"sense": "minimize", "coefficients": [1]},
                    "variables": [{"name": "x", "lower": None, "upper": None}],
                    "constraints": [],
                }
            },
        )
    )
    assert not result.succeeded
    assert result.evidence[0].supports_claim == SupportStatus.CONTRADICTS


def test_missing_context_fails_cleanly():
    tool = LinearOptimizerTool()
    result = tool.run(ToolRequest(claim=_claim()))
    assert not result.succeeded
    assert "context['optimization']" in (result.error or "")


def test_unsupported_kind_rejected():
    tool = LinearOptimizerTool()
    result = tool.run(
        ToolRequest(
            claim=_claim(),
            context={"optimization": {"kind": "convex", "objective": {}, "variables": []}},
        )
    )
    assert not result.succeeded
    assert "linear" in (result.error or "").lower()


def test_dimension_mismatch_rejected():
    tool = LinearOptimizerTool()
    result = tool.run(
        ToolRequest(
            claim=_claim(),
            context={
                "optimization": {
                    "objective": {"sense": "minimize", "coefficients": [1, 2]},
                    "variables": [{"name": "x", "lower": 0}],  # only 1 variable
                    "constraints": [],
                }
            },
        )
    )
    assert not result.succeeded
    assert "expected 1" in (result.error or "")


def test_invalid_constraint_sense_rejected():
    tool = LinearOptimizerTool()
    result = tool.run(
        ToolRequest(
            claim=_claim(),
            context={
                "optimization": {
                    "objective": {"sense": "minimize", "coefficients": [1]},
                    "variables": [{"name": "x", "lower": 0}],
                    "constraints": [
                        {"coefficients": [1], "sense": "<<", "rhs": 1}
                    ],
                }
            },
        )
    )
    assert not result.succeeded
    assert "<<" in (result.error or "")
