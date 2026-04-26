"""AnthropicCodeExecutor tests using a stubbed Anthropic client (no real API)."""

from __future__ import annotations

from types import SimpleNamespace

from factuality_harness.domain.claims import Claim
from factuality_harness.domain.epistemic_types import EpistemicType
from factuality_harness.domain.evidence import SourceType, SupportStatus
from factuality_harness.infrastructure.tools.base import ToolRequest
from factuality_harness.infrastructure.tools.python_executor_anthropic import (
    AnthropicCodeExecutor,
)


def _claim(text: str = "compute") -> Claim:
    return Claim(
        text=text,
        parent_question="x",
        epistemic_type=EpistemicType.NUMERICAL,
    )


def _success_response(stdout: str, stderr: str = "", return_code: int = 0):
    return SimpleNamespace(
        model="claude-opus-4-7",
        content=[
            SimpleNamespace(
                type="bash_code_execution_tool_result",
                content=SimpleNamespace(
                    type="bash_code_execution_result",
                    return_code=return_code,
                    stdout=stdout,
                    stderr=stderr,
                ),
            )
        ],
    )


def _no_tool_response(text: str = "Sure, I'll help with that."):
    return SimpleNamespace(
        model="claude-opus-4-7",
        content=[SimpleNamespace(type="text", text=text)],
    )


class _StubMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _StubClient:
    def __init__(self, response):
        self.messages = _StubMessages(response)


# ---------------------------------------------------------------------------


def test_runs_supplied_code_and_captures_stdout():
    client = _StubClient(_success_response(stdout="5\n"))
    tool = AnthropicCodeExecutor(client=client)
    result = tool.run(ToolRequest(claim=_claim(), context={"code": "print(2 + 3)"}))

    assert result.succeeded
    ev = result.evidence[0]
    assert ev.source_type == SourceType.COMPUTATION
    assert ev.source_name == "anthropic_code_execution"
    assert ev.supports_claim == SupportStatus.SUPPORTS
    assert "5" in ev.normalized_result["stdout"]
    assert ev.normalized_result["return_code"] == 0
    # The system prompt must require the model to call the tool.
    assert "code_execution" in client.messages.last_kwargs["tools"][0]["name"]


def test_nonzero_return_code_marks_evidence_insufficient():
    client = _StubClient(_success_response(stdout="", stderr="boom", return_code=2))
    tool = AnthropicCodeExecutor(client=client)
    result = tool.run(ToolRequest(claim=_claim(), context={"code": "raise SystemExit(2)"}))

    assert not result.succeeded
    assert result.evidence[0].supports_claim == SupportStatus.INSUFFICIENT
    assert result.evidence[0].normalized_result["return_code"] == 2


def test_model_response_without_tool_call_returns_error():
    client = _StubClient(_no_tool_response())
    tool = AnthropicCodeExecutor(client=client)
    result = tool.run(ToolRequest(claim=_claim(), context={"code": "print('x')"}))
    assert not result.succeeded
    assert "did not invoke" in (result.error or "")


def test_api_failure_propagates_as_clean_error():
    class _BoomMessages:
        def create(self, **kwargs):
            raise RuntimeError("503")

    client = SimpleNamespace(messages=_BoomMessages())
    tool = AnthropicCodeExecutor(client=client)
    result = tool.run(ToolRequest(claim=_claim(), context={"code": "x = 1"}))
    assert not result.succeeded
    assert "Anthropic API call failed" in (result.error or "")


def test_extracts_code_from_fenced_block_in_claim():
    client = _StubClient(_success_response(stdout="hello\n"))
    tool = AnthropicCodeExecutor(client=client)
    result = tool.run(
        ToolRequest(claim=_claim("Try this: ```python\nprint('hello')\n```"))
    )
    assert result.succeeded


def test_no_code_fails_cleanly():
    client = _StubClient(_success_response(stdout=""))
    tool = AnthropicCodeExecutor(client=client)
    result = tool.run(ToolRequest(claim=_claim("just prose, no code")))
    assert not result.succeeded
    assert "context['code']" in (result.error or "")


def test_platform_error_block_returned_as_error():
    response = SimpleNamespace(
        model="claude-opus-4-7",
        content=[
            SimpleNamespace(
                type="bash_code_execution_tool_result",
                content=SimpleNamespace(
                    type="bash_code_execution_tool_result_error",
                    error_code="sandbox_unavailable",
                ),
            )
        ],
    )
    tool = AnthropicCodeExecutor(client=_StubClient(response))
    result = tool.run(ToolRequest(claim=_claim(), context={"code": "print(1)"}))
    assert not result.succeeded
    assert "sandbox_unavailable" in (result.error or "")
