"""Construction helpers shared by the API and CLI.

Centralising construction here keeps the wiring consistent between entry points
and gives one place to swap real adapters in (LLM, retriever, persistence).

Environment variables that change defaults:

  FACTUALITY_HARNESS_LLM_DECOMPOSER   enable LLM-backed claim decomposition
                                      (1/true/yes). Off by default — keeps the
                                      harness deterministic unless explicitly opted in.
  FACTUALITY_HARNESS_LLM_TRANSLATOR   enable LLM-assisted tool-input translation
                                      (turns "did X cause Y?" + raw data into a
                                      causal_inference / forecast / sql payload).
                                      Off by default.
  FACTUALITY_HARNESS_LLM_CONTRADICTION enable LLM-backed NLI contradiction
                                      detection (catches non-lexical conflicts
                                      the antonym table misses). Off by default.
  FACTUALITY_HARNESS_PYTHON_EXECUTOR  backend for code execution evidence.
                                      One of: local (default), anthropic, e2b,
                                      self_hosted.
  ANTHROPIC_API_KEY                   preferred LLM for all LLM-backed paths.
  OPENAI_API_KEY                      fallback LLM.
  E2B_API_KEY                         required if FACTUALITY_HARNESS_PYTHON_EXECUTOR=e2b.
  FACTUALITY_HARNESS_AUDIT_DIR        directory for JSON audit traces.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..application.claim_decomposer import (
    ClaimDecomposer,
    RuleBasedClaimDecomposer,
)
from ..application.contradiction_checker import (
    ContradictionDetector,
    LLMContradictionDetector,
)
from ..application.llm_claim_decomposer import LLMClaimDecomposer
from ..application.module_registry import ModuleRegistry
from ..application.pipeline import FactualityPipeline
from ..application.tool_input_translator import (
    LLMToolInputTranslator,
    ToolInputTranslator,
)
from ..infrastructure.llm.base import LLM
from ..infrastructure.tools.base import Tool
from ..infrastructure.storage.repository import (
    AuditRepository,
    JsonAuditRepository,
)
from ..modules import (
    CodeModule,
    FinanceModule,
    GeneralModule,
    MarketingModule,
    MathModule,
    PolicyModule,
)


_TRUTHY = {"1", "true", "yes", "on"}


def build_module_registry() -> ModuleRegistry:
    return ModuleRegistry(
        modules=[
            GeneralModule(),
            MathModule(),
            PolicyModule(),
            # The following are DRAFT and so won't influence final answers,
            # but are surfaced in /modules for visibility.
            FinanceModule(),
            MarketingModule(),
            CodeModule(),
        ]
    )


def build_audit_repo() -> AuditRepository:
    audit_dir = os.environ.get("FACTUALITY_HARNESS_AUDIT_DIR", "./audit_logs")
    Path(audit_dir).mkdir(parents=True, exist_ok=True)
    return JsonAuditRepository(audit_dir)


def _build_llm_for_decomposition() -> LLM | None:
    """Construct an LLM adapter from whichever provider is configured.

    Anthropic preferred (the harness defaults to ``claude-opus-4-7``). Falls
    back to OpenAI if Anthropic isn't set. Returns ``None`` if no key is
    available — the caller then keeps the rule-based decomposer.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        from ..infrastructure.llm.anthropic_adapter import AnthropicAdapter

        try:
            return AnthropicAdapter()
        except Exception:
            return None
    if os.environ.get("OPENAI_API_KEY"):
        from ..infrastructure.llm.openai_adapter import OpenAIAdapter

        try:
            return OpenAIAdapter()
        except Exception:
            return None
    return None


def build_decomposer() -> ClaimDecomposer | None:
    """Return an LLM-backed decomposer if explicitly opted in and a provider
    is configured; otherwise ``None`` (callers fall back to rule-based).

    Returning ``None`` rather than an instance lets ``FactualityPipeline``
    use its own default — keeps a single source of truth for the fallback.
    """
    flag = os.environ.get("FACTUALITY_HARNESS_LLM_DECOMPOSER", "").strip().lower()
    if flag not in _TRUTHY:
        return None

    llm = _build_llm_for_decomposition()
    if llm is None:
        return None

    return LLMClaimDecomposer(llm=llm, fallback=RuleBasedClaimDecomposer())


def build_tool_input_translator() -> ToolInputTranslator | None:
    """Same opt-in pattern as the decomposer. Off by default."""
    flag = os.environ.get("FACTUALITY_HARNESS_LLM_TRANSLATOR", "").strip().lower()
    if flag not in _TRUTHY:
        return None
    llm = _build_llm_for_decomposition()
    if llm is None:
        return None
    return LLMToolInputTranslator(llm=llm)


_PYTHON_EXECUTOR_BACKENDS = {"local", "anthropic", "e2b", "self_hosted"}


def build_python_executor() -> Tool | None:
    """Return a non-default Python executor when explicitly selected.

    Returning ``None`` lets ``FactualityPipeline`` keep its built-in
    ``LocalSubprocessPythonExecutor``. Other backends are constructed
    lazily so the factory itself stays light.
    """
    backend = os.environ.get("FACTUALITY_HARNESS_PYTHON_EXECUTOR", "local").strip().lower()
    if backend not in _PYTHON_EXECUTOR_BACKENDS:
        return None
    if backend == "local":
        return None  # pipeline default

    if backend == "anthropic":
        # Requires ANTHROPIC_API_KEY at run time, but constructing the
        # adapter without one is allowed (the SDK reads env vars itself).
        from ..infrastructure.tools.python_executor_anthropic import (
            AnthropicCodeExecutor,
        )

        try:
            return AnthropicCodeExecutor()
        except Exception:
            return None

    if backend == "e2b":
        from ..infrastructure.tools.python_executor_e2b import E2BPythonExecutor

        return E2BPythonExecutor()

    if backend == "self_hosted":
        from ..infrastructure.tools.python_executor_self_hosted import (
            SelfHostedPythonExecutorStub,
        )

        return SelfHostedPythonExecutorStub()

    return None


def build_contradiction_detector() -> ContradictionDetector | None:
    """Build an LLM-backed NLI contradiction detector when explicitly enabled
    AND a provider key is configured. Returns ``None`` otherwise so the
    pipeline keeps its deterministic lexical default."""
    flag = os.environ.get("FACTUALITY_HARNESS_LLM_CONTRADICTION", "").strip().lower()
    if flag not in _TRUTHY:
        return None
    llm = _build_llm_for_decomposition()
    if llm is None:
        return None
    return LLMContradictionDetector(llm=llm)


def build_pipeline() -> FactualityPipeline:
    python_executor = build_python_executor()
    tool_overrides: dict[str, Tool] = {}
    if python_executor is not None:
        tool_overrides["python_executor"] = python_executor

    return FactualityPipeline(
        decomposer=build_decomposer(),  # may be None -> pipeline default
        tool_input_translator=build_tool_input_translator(),  # may be None
        contradiction_detector=build_contradiction_detector(),  # may be None
        module_registry=build_module_registry(),
        audit_repo=build_audit_repo(),
        tools=tool_overrides or None,
    )
