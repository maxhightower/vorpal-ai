from __future__ import annotations

from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

from .epistemic_types import EpistemicType


class ClaimStatus(str, Enum):
    PROPOSED = "PROPOSED"
    CLASSIFIED = "CLASSIFIED"
    ROUTED = "ROUTED"
    EVIDENCE_GATHERED = "EVIDENCE_GATHERED"
    VERDICT_ASSIGNED = "VERDICT_ASSIGNED"
    REJECTED = "REJECTED"


def _new_id() -> str:
    return f"claim_{uuid4().hex[:12]}"


class Claim(BaseModel):
    """An atomic, verifiable assertion derived from a question or a draft answer."""

    id: str = Field(default_factory=_new_id)
    text: str
    normalized_text: str | None = None
    parent_question: str
    epistemic_type: EpistemicType = EpistemicType.UNKNOWN
    domain: str | None = None
    requires_current_info: bool = False
    requires_computation: bool = False
    requires_formal_verification: bool = False
    requires_causal_inference: bool = False
    assumptions: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: ClaimStatus = ClaimStatus.PROPOSED

    def with_status(self, status: ClaimStatus) -> "Claim":
        return self.model_copy(update={"status": status})
