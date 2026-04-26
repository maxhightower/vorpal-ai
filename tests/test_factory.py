from __future__ import annotations

import pytest

from factuality_harness.application.llm_claim_decomposer import LLMClaimDecomposer
from factuality_harness.interfaces.factory import build_decomposer, build_pipeline


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Reset all relevant env vars per test so order doesn't matter."""
    for var in (
        "FACTUALITY_HARNESS_LLM_DECOMPOSER",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
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
