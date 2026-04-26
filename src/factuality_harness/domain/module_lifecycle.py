"""Data types for the module-development lifecycle.

These are the "what" of the lifecycle (gaps, proposals, gates, decisions);
the "how" lives in ``application/module_lifecycle.py``.

The spec's safety boundary is encoded here in two places:

 - ``PromotionDecision.requires_human_approval`` defaults to True. Even
   when ``promoted == True`` the registry must check this field before
   actually flipping a module to ACTIVE.
 - ``PromotionGate`` declares the threshold values; the gate is the only
   place where promotion-worthiness is decided. Code that promotes
   modules outside this gate is a bug.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from .epistemic_types import EpistemicType
from .evidence import EvidenceRequirement
from .modules import ModuleSpec


# ---------------------------------------------------------------------------
# Stage outputs
# ---------------------------------------------------------------------------


class DomainGap(BaseModel):
    """A domain where the harness is underperforming or unrepresented."""

    name: str
    description: str
    representative_audit_ids: list[str] = Field(default_factory=list)
    representative_queries: list[str] = Field(default_factory=list)
    severity: float = 0.0  # 0..1; higher = worse coverage gap


class ClaimCategory(BaseModel):
    """One category in a proposed module's claim taxonomy."""

    name: str
    description: str
    epistemic_type: EpistemicType
    example_queries: list[str] = Field(default_factory=list)


class SourceProposal(BaseModel):
    """A data source the module would draw evidence from.

    ``is_existing`` distinguishes "use the calculator we already have" from
    "we'd need to build a new connector"; the lifecycle pipeline can only
    deploy modules whose sources are all existing.
    """

    source_id: str
    description: str
    tool_kind: str  # "tool" | "retriever" | "module"
    is_existing: bool


class RouteProposal(BaseModel):
    """How a claim category should be routed to tools."""

    claim_category: str
    tool_names: list[str]


class ModuleProposal(BaseModel):
    """The complete output of stages 1-7. Inputs to evaluation + deployment."""

    spec: ModuleSpec
    gap: DomainGap
    taxonomy: list[ClaimCategory] = Field(default_factory=list)
    sources: list[SourceProposal] = Field(default_factory=list)
    routes: list[RouteProposal] = Field(default_factory=list)
    evidence_requirements: list[EvidenceRequirement] = Field(default_factory=list)
    benchmark_case_names: list[str] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def is_deployable(self) -> bool:
        """True iff every proposed source is already wired in.

        We refuse to advance proposals that depend on tools we don't have —
        the alternative would be silently failing tool calls in shadow.
        """
        return bool(self.sources) and all(s.is_existing for s in self.sources)


# ---------------------------------------------------------------------------
# Shadow execution + metrics
# ---------------------------------------------------------------------------


class ShadowVerdict(BaseModel):
    """A SHADOW module's verdict on a claim, recorded for later comparison.

    These are written to the audit trace alongside the active verdict but
    NEVER affect the final answer.
    """

    module_name: str
    module_version: str
    claim_id: str
    proposed_verdict: str  # Verdict.value
    proposed_confidence: str  # ConfidenceLevel.value
    rationale: str
    agrees_with_active: bool


class ShadowMetrics(BaseModel):
    """Aggregate metrics for one SHADOW module over many runs.

    ``total_verdicts`` is the denominator for every rate. ``agreement_rate``
    is a sanity check (does the shadow module disagree wildly with the
    active path?) — by itself it's not a quality signal because the active
    path may be wrong; pair it with an eval-suite pass rate for the real
    promotion gate.
    """

    module_name: str
    module_version: str
    total_runs: int = 0
    total_verdicts: int = 0
    agreements: int = 0
    disagreements: int = 0

    @property
    def agreement_rate(self) -> float:
        return self.agreements / self.total_verdicts if self.total_verdicts else 0.0


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------


class PromotionGate(BaseModel):
    """Thresholds a module must clear to be eligible for ACTIVE.

    Defaults are deliberately strict. ``min_runs`` prevents a single happy
    test run from auto-promoting; ``min_eval_pass_rate`` ensures the
    benchmark cases actually exist and pass.
    """

    min_eval_pass_rate: float = 0.9
    max_overclaim_rate: float = 0.05
    min_classification_accuracy: float = 0.8
    min_shadow_runs: int = 20
    min_shadow_agreement_rate: float = 0.7


class PromotionDecision(BaseModel):
    """Output of ``promote_if_passes_thresholds``.

    The registry must check ``requires_human_approval`` before actually
    flipping the module's status. This is the spec's safety boundary —
    auto-promotion without human approval is forbidden by default.
    """

    module_name: str
    module_version: str
    eligible: bool
    reasons: list[str] = Field(default_factory=list)
    requires_human_approval: bool = True
    promoted: bool = False  # set True only after the registry actually flips it
    decided_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
