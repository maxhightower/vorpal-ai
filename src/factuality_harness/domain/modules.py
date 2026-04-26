from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ModuleStatus(str, Enum):
    """Lifecycle for domain modules. Only ACTIVE may influence final answers."""

    DRAFT = "draft"
    SHADOW = "shadow"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class ModuleSpec(BaseModel):
    """Declarative specification for a domain module.

    The actual behavior lives in subclasses of ``modules.base.BaseDomainModule``;
    this spec is what the registry stores and exposes via the API.
    """

    name: str
    description: str
    version: str = "0.1.0"
    status: ModuleStatus = ModuleStatus.DRAFT
    claim_types: list[str] = Field(default_factory=list)
    source_policy: dict = Field(default_factory=dict)
    tool_policy: dict = Field(default_factory=dict)
    evidence_standards: dict = Field(default_factory=dict)
    benchmark_cases: list[dict] = Field(default_factory=list)
