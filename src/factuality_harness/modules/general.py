"""GeneralModule: ordinary factual Q&A.

Defaults to applying to everything at low relevance so it is the fallback
module. Activated by default in the registry.
"""

from __future__ import annotations

from ..domain.claims import Claim
from ..domain.evidence import EvidenceRequirement, SourceQuality, SourceType
from ..domain.epistemic_types import EpistemicType
from ..domain.modules import ModuleSpec, ModuleStatus
from .base import BaseDomainModule


class GeneralModule(BaseDomainModule):
    def __init__(self) -> None:
        super().__init__(
            ModuleSpec(
                name="general",
                description="Default module for ordinary factual questions.",
                version="0.1.0",
                status=ModuleStatus.ACTIVE,
                claim_types=[t.value for t in EpistemicType],
                evidence_standards={
                    "DIRECT_FACT": "Requires retrieval source.",
                    "INTERPRETIVE": "Must surface assumptions.",
                },
            )
        )

    def applies_to(self, question: str, claims: list[Claim]) -> float:
        return 0.1

    def required_evidence_for(self, claim: Claim) -> list[EvidenceRequirement]:
        if claim.epistemic_type == EpistemicType.DIRECT_FACT:
            return [
                EvidenceRequirement(
                    claim_type=claim.epistemic_type,
                    required_source_types=[
                        SourceType.LOCAL_DOCUMENT,
                        SourceType.WEB_PAGE,
                        SourceType.STRUCTURED_DATA,
                    ],
                    minimum_source_quality=SourceQuality.SECONDARY,
                    requires_date_check=claim.requires_current_info,
                )
            ]
        return []
