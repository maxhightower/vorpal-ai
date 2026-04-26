"""Construction helpers shared by the API and CLI.

Centralising construction here keeps the wiring consistent between entry points
and gives one place to swap real adapters in (LLM, retriever, persistence).

Environment variables that change defaults:

  FACTUALITY_HARNESS_LLM_DECOMPOSER  enable LLM-backed claim decomposition
                                     (1/true/yes). Off by default — keeps the
                                     harness deterministic unless explicitly opted in.
  ANTHROPIC_API_KEY                  preferred LLM for decomposition.
  OPENAI_API_KEY                     fallback LLM for decomposition.
  FACTUALITY_HARNESS_AUDIT_DIR       directory for JSON audit traces.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..application.claim_decomposer import (
    ClaimDecomposer,
    RuleBasedClaimDecomposer,
)
from ..application.llm_claim_decomposer import LLMClaimDecomposer
from ..application.module_registry import ModuleRegistry
from ..application.pipeline import FactualityPipeline
from ..infrastructure.llm.base import LLM
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


def build_pipeline() -> FactualityPipeline:
    return FactualityPipeline(
        decomposer=build_decomposer(),  # may be None -> pipeline default
        module_registry=build_module_registry(),
        audit_repo=build_audit_repo(),
    )
