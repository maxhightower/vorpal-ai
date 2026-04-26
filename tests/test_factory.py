from __future__ import annotations

import pytest

from factuality_harness.application.contradiction_checker import (
    LexicalContradictionDetector,
    LLMContradictionDetector,
)
from factuality_harness.application.llm_claim_decomposer import LLMClaimDecomposer
from factuality_harness.application.tool_input_translator import (
    LLMToolInputTranslator,
    NullToolInputTranslator,
)
from factuality_harness.infrastructure.tools.python_executor import (
    LocalSubprocessPythonExecutor,
)
from factuality_harness.infrastructure.tools.python_executor_anthropic import (
    AnthropicCodeExecutor,
)
from factuality_harness.infrastructure.tools.python_executor_e2b import (
    E2BPythonExecutor,
)
from factuality_harness.infrastructure.tools.python_executor_self_hosted import (
    SelfHostedPythonExecutorStub,
)
from factuality_harness.interfaces.factory import (
    build_contradiction_detector,
    build_decomposer,
    build_pipeline,
    build_python_executor,
    build_tool_input_translator,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Reset all relevant env vars per test so order doesn't matter."""
    for var in (
        "FACTUALITY_HARNESS_LLM_DECOMPOSER",
        "FACTUALITY_HARNESS_LLM_TRANSLATOR",
        "FACTUALITY_HARNESS_LLM_CONTRADICTION",
        "FACTUALITY_HARNESS_PYTHON_EXECUTOR",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "E2B_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_build_decomposer_returns_none_by_default():
    # No env var set -> fall back to pipeline's default rule-based decomposer.
    assert build_decomposer() is None


def test_build_decomposer_flag_without_keys_returns_none(monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_LLM_DECOMPOSER", "1")
    # No API keys -> graceful fallback.
    assert build_decomposer() is None


def test_build_decomposer_uses_anthropic_when_keyed(monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_LLM_DECOMPOSER", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    decomposer = build_decomposer()
    assert isinstance(decomposer, LLMClaimDecomposer)
    # Inspect the wrapped LLM type via duck-typing.
    assert decomposer._llm.__class__.__name__ == "AnthropicAdapter"


def test_build_decomposer_falls_back_to_openai(monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_LLM_DECOMPOSER", "yes")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    decomposer = build_decomposer()
    assert isinstance(decomposer, LLMClaimDecomposer)
    assert decomposer._llm.__class__.__name__ == "OpenAIAdapter"


def test_build_decomposer_anthropic_preferred_over_openai(monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_LLM_DECOMPOSER", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    decomposer = build_decomposer()
    assert decomposer is not None
    assert decomposer._llm.__class__.__name__ == "AnthropicAdapter"


def test_build_pipeline_works_without_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_AUDIT_DIR", str(tmp_path))
    pipeline = build_pipeline()
    # The pipeline's decomposer is its own default (RuleBasedClaimDecomposer).
    assert pipeline.decomposer.__class__.__name__ == "RuleBasedClaimDecomposer"


def test_build_pipeline_uses_llm_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("FACTUALITY_HARNESS_LLM_DECOMPOSER", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    pipeline = build_pipeline()
    assert isinstance(pipeline.decomposer, LLMClaimDecomposer)


def test_build_tool_input_translator_default_returns_none():
    assert build_tool_input_translator() is None


def test_build_tool_input_translator_flag_without_keys_returns_none(monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_LLM_TRANSLATOR", "1")
    assert build_tool_input_translator() is None


def test_build_tool_input_translator_with_anthropic_key(monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_LLM_TRANSLATOR", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    translator = build_tool_input_translator()
    assert isinstance(translator, LLMToolInputTranslator)


def test_build_pipeline_uses_translator_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("FACTUALITY_HARNESS_LLM_TRANSLATOR", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    pipeline = build_pipeline()
    assert isinstance(pipeline.tool_input_translator, LLMToolInputTranslator)


def test_build_pipeline_default_translator_is_null(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_AUDIT_DIR", str(tmp_path))
    pipeline = build_pipeline()
    assert isinstance(pipeline.tool_input_translator, NullToolInputTranslator)


def test_build_contradiction_detector_default_returns_none():
    assert build_contradiction_detector() is None


def test_build_contradiction_detector_flag_without_keys_returns_none(monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_LLM_CONTRADICTION", "1")
    assert build_contradiction_detector() is None


def test_build_contradiction_detector_with_anthropic_key(monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_LLM_CONTRADICTION", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    detector = build_contradiction_detector()
    assert isinstance(detector, LLMContradictionDetector)


def test_build_pipeline_default_contradiction_detector_is_lexical(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_AUDIT_DIR", str(tmp_path))
    pipeline = build_pipeline()
    assert isinstance(pipeline.contradiction_detector, LexicalContradictionDetector)


def test_build_pipeline_uses_llm_contradiction_detector_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("FACTUALITY_HARNESS_LLM_CONTRADICTION", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    pipeline = build_pipeline()
    assert isinstance(pipeline.contradiction_detector, LLMContradictionDetector)


# ---------------------------------------------------------------------------
# Python-executor backend selection
# ---------------------------------------------------------------------------


def test_build_python_executor_default_returns_none():
    # Unset / "local" both mean "use the pipeline's built-in executor".
    assert build_python_executor() is None


def test_build_python_executor_unknown_backend_returns_none(monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_PYTHON_EXECUTOR", "made_up")
    assert build_python_executor() is None


def test_build_python_executor_anthropic(monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_PYTHON_EXECUTOR", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert isinstance(build_python_executor(), AnthropicCodeExecutor)


def test_build_python_executor_e2b(monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_PYTHON_EXECUTOR", "e2b")
    assert isinstance(build_python_executor(), E2BPythonExecutor)


def test_build_python_executor_self_hosted(monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_PYTHON_EXECUTOR", "self_hosted")
    assert isinstance(build_python_executor(), SelfHostedPythonExecutorStub)


def test_pipeline_default_python_executor_is_local(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_AUDIT_DIR", str(tmp_path))
    pipeline = build_pipeline()
    tools = pipeline.evidence_builder._tools
    assert isinstance(tools["python_executor"], LocalSubprocessPythonExecutor)


def test_pipeline_uses_e2b_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("FACTUALITY_HARNESS_PYTHON_EXECUTOR", "e2b")
    pipeline = build_pipeline()
    tools = pipeline.evidence_builder._tools
    assert isinstance(tools["python_executor"], E2BPythonExecutor)


def test_pipeline_uses_self_hosted_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTUALITY_HARNESS_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("FACTUALITY_HARNESS_PYTHON_EXECUTOR", "self_hosted")
    pipeline = build_pipeline()
    tools = pipeline.evidence_builder._tools
    assert isinstance(tools["python_executor"], SelfHostedPythonExecutorStub)
