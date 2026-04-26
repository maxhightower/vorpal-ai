"""E2BPythonExecutor tests using an injected fake sandbox factory."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from factuality_harness.domain.claims import Claim
from factuality_harness.domain.epistemic_types import EpistemicType
from factuality_harness.domain.evidence import SourceType, SupportStatus
from factuality_harness.infrastructure.tools.base import ToolRequest
from factuality_harness.infrastructure.tools.python_executor_e2b import (
    E2BPythonExecutor,
)


def _claim(text: str = "compute") -> Claim:
    return Claim(
        text=text,
        parent_question="x",
        epistemic_type=EpistemicType.NUMERICAL,
    )


def _make_fake_sandbox(stdout=None, stderr=None, error=None, raise_on_run=None):
    stdout = stdout or []
    stderr = stderr or []

    class _FakeSandbox:
        def __init__(self, *_, **__):
            self.killed = False

        def run_code(self, code, *, timeout=None):
            if raise_on_run:
                raise raise_on_run
            return SimpleNamespace(
                logs=SimpleNamespace(stdout=stdout, stderr=stderr),
                error=error,
            )

        def kill(self):
            self.killed = True

    return _FakeSandbox


# ---------------------------------------------------------------------------


def test_successful_execution_emits_evidence():
    factory = _make_fake_sandbox(stdout=["hello", "world"])
    tool = E2BPythonExecutor(api_key="test", sandbox_factory=factory)
    result = tool.run(ToolRequest(claim=_claim(), context={"code": "print('hi')"}))
    assert result.succeeded
    ev = result.evidence[0]
    assert ev.source_type == SourceType.COMPUTATION
    assert ev.supports_claim == SupportStatus.SUPPORTS
    assert "hello" in ev.normalized_result["stdout"]
    assert ev.normalized_result["backend"] == "e2b"


def test_runtime_error_marks_insufficient():
    factory = _make_fake_sandbox(
        stdout=[],
        stderr=["Traceback ..."],
        error=SimpleNamespace(name="ZeroDivisionError", value="division by zero"),
    )
    tool = E2BPythonExecutor(api_key="test", sandbox_factory=factory)
    result = tool.run(ToolRequest(claim=_claim(), context={"code": "1/0"}))
    assert not result.succeeded
    assert result.evidence[0].supports_claim == SupportStatus.INSUFFICIENT
    assert "ZeroDivisionError" in result.evidence[0].normalized_result["stderr"]


def test_run_code_exception_returns_error():
    factory = _make_fake_sandbox(raise_on_run=RuntimeError("kaboom"))
    tool = E2BPythonExecutor(api_key="test", sandbox_factory=factory)
    result = tool.run(ToolRequest(claim=_claim(), context={"code": "x = 1"}))
    assert not result.succeeded
    assert "kaboom" in (result.error or "")


def test_missing_api_key_fails_cleanly(monkeypatch):
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    # api_key is None and the env var is absent, so the executor should
    # refuse before constructing a sandbox.
    tool = E2BPythonExecutor(sandbox_factory=_make_fake_sandbox())
    result = tool.run(ToolRequest(claim=_claim(), context={"code": "x = 1"}))
    assert not result.succeeded
    assert "E2B_API_KEY" in (result.error or "")


def test_missing_sdk_returns_error_when_no_factory_supplied(monkeypatch):
    """Force the lazy import path to fail cleanly. We don't have a way to
    universally remove the e2b_code_interpreter module (it may not even be
    installed in this env), so we monkeypatch the resolver to simulate it."""
    tool = E2BPythonExecutor(api_key="test")
    monkeypatch.setattr(tool, "_resolve_sandbox_factory", lambda: None)
    result = tool.run(ToolRequest(claim=_claim(), context={"code": "x = 1"}))
    assert not result.succeeded
    assert "e2b_code_interpreter" in (result.error or "")


def test_no_code_fails_cleanly():
    tool = E2BPythonExecutor(api_key="test", sandbox_factory=_make_fake_sandbox())
    result = tool.run(ToolRequest(claim=_claim("just prose")))
    assert not result.succeeded
    assert "context['code']" in (result.error or "")


def test_extracts_code_from_fenced_block():
    factory = _make_fake_sandbox(stdout=["hello"])
    tool = E2BPythonExecutor(api_key="test", sandbox_factory=factory)
    result = tool.run(
        ToolRequest(claim=_claim("Try ```python\nprint('hello')\n```"))
    )
    assert result.succeeded


def test_sandbox_killed_after_run():
    factory = _make_fake_sandbox(stdout=["ok"])
    tool = E2BPythonExecutor(api_key="test", sandbox_factory=factory)
    # Cleanup is fire-and-forget; the test confirms run() doesn't blow up
    # when ``.kill()`` exists on the sandbox.
    result = tool.run(ToolRequest(claim=_claim(), context={"code": "x=1"}))
    assert result.succeeded
