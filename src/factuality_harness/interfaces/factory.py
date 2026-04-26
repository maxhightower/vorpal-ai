"""Construction helpers shared by the API and CLI.

Centralising construction here keeps the wiring consistent between entry points
and gives one place to swap real adapters in (LLM, retriever, persistence).
"""

from __future__ import annotations

import os
from pathlib import Path

from ..application.module_registry import ModuleRegistry
from ..application.pipeline import FactualityPipeline
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


def build_pipeline() -> FactualityPipeline:
    return FactualityPipeline(
        module_registry=build_module_registry(),
        audit_repo=build_audit_repo(),
    )
