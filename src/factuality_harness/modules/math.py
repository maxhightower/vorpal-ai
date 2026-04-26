"""MathModule: arithmetic and deterministic computation."""

from __future__ import annotations

from ..domain.claims import Claim
from ..domain.epistemic_types import EpistemicType
from ..domain.evidence import EvidenceRequirement, SourceQuality, SourceType
from ..domain.modules import ModuleSpec, ModuleStatus
from .base import BaseDomainModule


class MathModule(BaseDomainModule):
    def __init__(self) -> None:
        super().__init__(
            ModuleSpec(
                name="math",
                description="Deterministic arithmetic and numeric computation.",
                version="0.1.0",
                status=ModuleStatus.ACTIVE,
                claim_types=[EpistemicType.NUMERICAL.value, EpistemicType.LOGICAL.value],
                tool_policy={"NUMERICAL": ["calculator"], "LOGICAL": ["theorem_prover"]},
                evidence_standards={
                    "NUMERICAL": "Must be computed or read from authoritative structured data.",
                },
            )
        )

    def applies_to(self, question: str, claims: list[Claim]) -> float:
        if any(c.epistemic_type == EpistemicType.NUMERICAL for c in claims):
            return 0.9
        return 0.0

    def required_evidence_for(self, claim: Claim) -> list[EvidenceRequirement]:
        if claim.epistemic_type == EpistemicType.NUMERICAL:
            return [
                EvidenceRequirement(
                    claim_type=claim.epistemic_type,
                    required_source_types=[SourceType.COMPUTATION, SourceType.STRUCTURED_DATA],
                    minimum_source_quality=SourceQuality.AUTHORITATIVE,
                    requires_computation=True,
                )
            ]
        return []
