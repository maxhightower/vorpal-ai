from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from ...domain.claims import Claim
from ...domain.evidence import (
    Evidence,
    FreshnessStatus,
    SourceQuality,
    SourceType,
    SupportStatus,
)


class ToolRequest(BaseModel):
    claim: Claim
    context: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    succeeded: bool = True
    evidence: list[Evidence] = Field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Tool(Protocol):
    """All verification tools share this contract.

    A tool *produces evidence*; it does NOT decide whether a claim is verified.
    Verdicts are decided by the verifier using the evidence the tool produces.
    """

    name: str

    def run(self, request: ToolRequest) -> ToolResult: ...


def build_evidence(
    *,
    claim_id: str,
    source_type: SourceType,
    source_name: str,
    quote_or_result: str,
    normalized_result: dict | None = None,
    supports_claim: SupportStatus = SupportStatus.INSUFFICIENT,
    source_quality: SourceQuality = SourceQuality.UNKNOWN,
    freshness: FreshnessStatus = FreshnessStatus.NOT_TIME_SENSITIVE,
    source_uri: str | None = None,
    notes: str | None = None,
) -> Evidence:
    """Convenience constructor used by tools to keep evidence creation uniform."""
    return Evidence(
        claim_id=claim_id,
        source_type=source_type,
        source_name=source_name,
        source_uri=source_uri,
        quote_or_result=quote_or_result,
        normalized_result=normalized_result,
        supports_claim=supports_claim,
        source_quality=source_quality,
        freshness=freshness,
        notes=notes,
    )
