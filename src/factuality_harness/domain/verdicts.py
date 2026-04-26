from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .confidence import ConfidenceLevel
from .evidence import EvidenceTable


class Verdict(str, Enum):
    VERIFIED = "VERIFIED"
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    COMPUTED = "COMPUTED"
    FORMALLY_PROVEN = "FORMALLY_PROVEN"
    UNCLEAR = "UNCLEAR"
    SPECULATIVE = "SPECULATIVE"


class ClaimVerdict(BaseModel):
    claim_id: str
    verdict: Verdict
    confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    answer: str
    confidence_summary: str
    unsupported_or_uncertain_claims: list[str] = Field(default_factory=list)
    evidence_table: EvidenceTable
    audit_id: str
