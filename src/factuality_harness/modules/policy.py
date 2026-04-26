"""PolicyModule: rule/procedure questions against loaded policy documents."""

from __future__ import annotations

from ..domain.claims import Claim
from ..domain.epistemic_types import EpistemicType
from ..domain.evidence import EvidenceRequirement, SourceQuality, SourceType
from ..domain.modules import ModuleSpec, ModuleStatus
from .base import BaseDomainModule


class PolicyModule(BaseDomainModule):
    def __init__(self) -> None:
        super().__init__(
            ModuleSpec(
                name="policy",
                description="Procedural compliance against loaded policy documents.",
                version="0.1.0",
                status=ModuleStatus.ACTIVE,
                claim_types=[EpistemicType.PROCEDURAL.value],
                tool_policy={"PROCEDURAL": ["rule_engine", "local_document_retriever"]},
                evidence_standards={
                    "PROCEDURAL": "Must reference applicable, dated policy text.",
                },
            )
        )

    def applies_to(self, question: str, claims: list[Claim]) -> float:
        if any(c.epistemic_type == EpistemicType.PROCEDURAL for c in claims):
            return 0.8
        return 0.0

    def required_evidence_for(self, claim: Claim) -> list[EvidenceRequirement]:
        if claim.epistemic_type == EpistemicType.PROCEDURAL:
            return [
                EvidenceRequirement(
                    claim_type=claim.epistemic_type,
                    required_source_types=[
                        SourceType.RULE_ENGINE,
                        SourceType.LOCAL_DOCUMENT,
                    ],
                    minimum_source_quality=SourceQuality.AUTHORITATIVE,
                    requires_date_check=True,
                    requires_contradiction_search=True,
                )
            ]
        return []
