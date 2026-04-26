from factuality_harness.domain.claims import Claim
from factuality_harness.domain.epistemic_types import EpistemicType
from factuality_harness.infrastructure.tools.base import ToolRequest
from factuality_harness.infrastructure.tools.calculator import CalculatorTool


def _claim(text: str) -> Claim:
    return Claim(
        text=text,
        parent_question=text,
        epistemic_type=EpistemicType.NUMERICAL,
    )


def test_percentage_change():
    tool = CalculatorTool()
    result = tool.run(ToolRequest(claim=_claim("percentage increase from 100 to 125")))
    assert result.succeeded
    assert len(result.evidence) == 1
    norm = result.evidence[0].normalized_result
    assert norm["kind"] == "percentage_change"
    assert abs(norm["percent_change"] - 25.0) < 1e-9


def test_arithmetic():
    tool = CalculatorTool()
    result = tool.run(ToolRequest(claim=_claim("compute 12 * 7")))
    assert result.succeeded
    assert result.evidence[0].normalized_result["result"] == 84


def test_division_by_zero_handled():
    tool = CalculatorTool()
    result = tool.run(ToolRequest(claim=_claim("compute 10 / 0")))
    assert not result.succeeded


def test_unrecognized_returns_failure():
    tool = CalculatorTool()
    result = tool.run(ToolRequest(claim=_claim("describe the moon")))
    assert not result.succeeded
