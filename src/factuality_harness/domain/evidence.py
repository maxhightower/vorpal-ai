from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .claims import Claim
from .epistemic_types import EpistemicType


class SourceType(str, Enum):
    LOCAL_DOCUMENT = "LOCAL_DOCUMENT"
    WEB_PAGE = "WEB_PAGE"
    STRUCTURED_DATA = "STRUCTURED_DATA"
    COMPUTATION = "COMPUTATION"
    RULE_ENGINE = "RULE_ENGINE"
    THEOREM_PROVER = "THEOREM_PROVER"
    CAUSAL_MODEL = "CAUSAL_MODEL"
    FORECAST_MODEL = "FORECAST_MODEL"
    OPTIMIZER = "OPTIMIZER"
    SIMULATOR = "SIMULATOR"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    LLM_ASSERTION = "LLM_ASSERTION"
    NONE = "NONE"


class SupportStatus(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    PARTIALLY_SUPPORTS = "PARTIALLY_SUPPORTS"
    IRRELEVANT = "IRRELEVANT"
    INSUFFICIENT = "INSUFFICIENT"


class SourceQuality(str, Enum):
    AUTHORITATIVE = "AUTHORITATIVE"
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    LOW_QUALITY = "LOW_QUALITY"
    UNKNOWN = "UNKNOWN"


class FreshnessStatus(str, Enum):
    CURRENT = "CURRENT"
    STALE = "STALE"
    UNDATED = "UNDATED"
    NOT_TIME_SENSITIVE = "NOT_TIME_SENSITIVE"


def _new_id() -> str:
    return f"ev_{uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Evidence(BaseModel):
    """A single piece of evidence retrieved or computed for a claim."""

    id: str = Field(default_factory=_new_id)
    claim_id: str
    source_type: SourceType
    source_name: str
    source_uri: str | None = None
    retrieved_at: datetime = Field(default_factory=_now)
    quote_or_result: str
    normalized_result: dict | None = None
    supports_claim: SupportStatus = SupportStatus.INSUFFICIENT
    source_quality: SourceQuality = SourceQuality.UNKNOWN
    freshness: FreshnessStatus = FreshnessStatus.UNDATED
    notes: str | None = None


class EvidenceRequirement(BaseModel):
    """Module-defined contract for what counts as adequate evidence for a claim."""

    claim_type: EpistemicType
    required_source_types: list[SourceType] = Field(default_factory=list)
    minimum_source_quality: SourceQuality = SourceQuality.SECONDARY
    requires_date_check: bool = False
    requires_computation: bool = False
    requires_contradiction_search: bool = True
    requires_human_review: bool = False
    rationale: str | None = None


class EvidenceTable(BaseModel):
    """The internal evidence table for a single run."""

    original_question: str
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    # Filled with ClaimVerdict objects. Typed as Any to avoid a circular import
    # between evidence.py and verdicts.py; the orchestrator inserts validated
    # ClaimVerdict instances and consumers should treat them as such.
    verdicts: list[Any] = Field(default_factory=list)

    def evidence_for(self, claim_id: str) -> list[Evidence]:
        return [e for e in self.evidence if e.claim_id == claim_id]

    def claim_by_id(self, claim_id: str) -> Claim | None:
        for c in self.claims:
            if c.id == claim_id:
                return c
        return None
