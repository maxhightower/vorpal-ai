from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .claims import Claim
from .catalog import DataCatalogEntry
from .evidence import Evidence
from .module_lifecycle import ShadowVerdict
from .verdicts import ClaimVerdict


def _new_audit_id() -> str:
    return f"audit_{uuid4().hex[:16]}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ToolCallRecord(BaseModel):
    tool_name: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    succeeded: bool = True
    error: str | None = None


class AuditTrace(BaseModel):
    audit_id: str = Field(default_factory=_new_audit_id)
    timestamp: datetime = Field(default_factory=_now)
    original_question: str
    decomposed_claims: list[Claim] = Field(default_factory=list)
    claim_classifications: dict[str, str] = Field(default_factory=dict)
    tools_called: list[ToolCallRecord] = Field(default_factory=list)
    # Data sources the discovery layer selected for this run. Recorded in
    # the audit trace so callers can see which connectors were touched
    # (and which were skipped) per request.
    data_sources_consulted: list[DataCatalogEntry] = Field(default_factory=list)
    retrieved_evidence: list[Evidence] = Field(default_factory=list)
    contradictions_found: list[str] = Field(default_factory=list)
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    # SHADOW modules never affect the final answer — their proposed verdicts
    # are recorded here for offline metric collection only.
    shadow_verdicts: list[ShadowVerdict] = Field(default_factory=list)
    draft_answer: str = ""
    final_answer: str = ""
    unsupported_claims_removed: list[str] = Field(default_factory=list)
    module_versions_used: dict[str, str] = Field(default_factory=dict)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
