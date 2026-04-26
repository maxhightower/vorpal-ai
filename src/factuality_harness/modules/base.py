"""BaseDomainModule.

A domain module declares:

 - which claims it applies to,
 - what evidence is required for those claims,
 - which tools the router should prefer for those claims,
 - and how to validate the resulting evidence.

The module lifecycle (DRAFT → SHADOW → ACTIVE → DEPRECATED) is enforced by the
``ModuleRegistry``. Only ACTIVE modules may influence final answers.
"""

from __future__ import annotations

from ..domain.claims import Claim
from ..domain.evidence import Evidence, EvidenceRequirement
from ..domain.modules import ModuleSpec, ModuleStatus
from ..domain.verdicts import ClaimVerdict


class BaseDomainModule:
    """Abstract base. Subclasses must set ``spec``."""

    spec: ModuleSpec

    def __init__(self, spec: ModuleSpec) -> None:
        self.spec = spec

    # Override in subclasses.
    def applies_to(self, question: str, claims: list[Claim]) -> float:
        """Return a relevance score in [0, 1] for this module to handle this query."""
        return 0.0

    def enrich_claims(self, claims: list[Claim]) -> list[Claim]:
        """Optionally tag claims with domain metadata (no I/O)."""
        return claims

    def required_evidence_for(self, claim: Claim) -> list[EvidenceRequirement]:
        return []

    def validate_evidence(
        self, claim: Claim, evidence: list[Evidence]
    ) -> ClaimVerdict | None:
        """Return a verdict only if this module wants to override the default."""
        return None

    def is_active(self) -> bool:
        return self.spec.status == ModuleStatus.ACTIVE
