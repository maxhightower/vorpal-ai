from __future__ import annotations

from factuality_harness.domain.claims import Claim
from factuality_harness.domain.epistemic_types import EpistemicType
from factuality_harness.domain.evidence import SourceType, SupportStatus
from factuality_harness.infrastructure.tools.base import ToolRequest
from factuality_harness.infrastructure.tools.python_executor import (
    LocalSubprocessPythonExecutor,
)


def _claim(text: str = "compute something") -> Claim:
    return Claim(
        text=text,
        parent_question="x",
        epistemic_type=EpistemicType.NUMERICAL,
    )


def test_runs_simple_code_and_captures_stdout():
    tool = LocalSubprocessPythonExecutor()
    req = ToolRequest(claim=_claim(), context={"code": "print(2 + 3)"})
    result = tool.run(req)
    assert result.succeeded
    ev = result.evidence[0]
    assert ev.source_type == SourceType.COMPUTATION
    assert ev.supports_claim == SupportStatus.SUPPORTS
    assert "5" in ev.normalized_result["stdout"]
    assert ev.normalized_result["return_code"] == 0


def test_nonzero_exit_marks_evidence_insufficient():
    tool = LocalSubprocessPythonExecutor()
    req = ToolRequest(
        claim=_claim(),
        context={"code": "raise SystemExit(2)"},
    )
    result = tool.run(req)
    assert not result.succeeded
    assert result.evidence[0].supports_claim == SupportStatus.INSUFFICIENT
    assert result.evidence[0].normalized_result["return_code"] == 2


def test_timeout_is_enforced():
    tool = LocalSubprocessPythonExecutor()
    req = ToolRequest(
        claim=_claim(),
        context={"code": "import time\ntime.sleep(5)", "timeout_s": 0.5},
    )
    result = tool.run(req)
    assert not result.succeeded
    assert "timed out" in (result.error or "")


def test_extracts_code_from_fenced_block_in_claim():
    tool = LocalSubprocessPythonExecutor()
    req = ToolRequest(
        claim=_claim(
            "Here is a snippet:\n```python\nprint('hello world')\n```"
        )
    )
    result = tool.run(req)
    assert result.succeeded
    assert "hello world" in result.evidence[0].normalized_result["stdout"]


def test_no_code_fails_cleanly():
    tool = LocalSubprocessPythonExecutor()
    result = tool.run(ToolRequest(claim=_claim("describe the moon")))
    assert not result.succeeded
    assert "context['code']" in (result.error or "")
