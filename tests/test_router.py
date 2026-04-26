from factuality_harness.application.router import ToolRouter
from factuality_harness.domain.claims import Claim
from factuality_harness.domain.epistemic_types import EpistemicType


def _claim(t: EpistemicType) -> Claim:
    return Claim(text="x", parent_question="x", epistemic_type=t)


def test_numerical_routes_to_calculator():
    assert "calculator" in ToolRouter().route(_claim(EpistemicType.NUMERICAL)).tool_names


def test_causal_routes_to_causal_model():
    assert "causal_model" in ToolRouter().route(_claim(EpistemicType.CAUSAL)).tool_names


def test_procedural_routes_to_rule_engine():
    tools = ToolRouter().route(_claim(EpistemicType.PROCEDURAL)).tool_names
    assert "rule_engine" in tools


def test_speculative_has_no_tools():
    assert ToolRouter().route(_claim(EpistemicType.SPECULATIVE)).tool_names == []
