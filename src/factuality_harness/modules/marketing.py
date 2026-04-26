"""MarketingModule (stub).

Causal marketing claims (lift, attribution) require experiment or causal-model
evidence; the module declares this contract but does not run yet.
"""

from __future__ import annotations

from ..domain.modules import ModuleSpec, ModuleStatus
from .base import BaseDomainModule


_MARKETING_CLAIM_TYPES = [
    "attribution",
    "lift",
    "CAC",
    "LTV",
    "conversion",
    "segment",
    "positioning",
    "retention",
]


class MarketingModule(BaseDomainModule):
    def __init__(self) -> None:
        super().__init__(
            ModuleSpec(
                name="marketing",
                description="Marketing analytics (stub — pending causal infra).",
                version="0.0.1",
                status=ModuleStatus.DRAFT,
                claim_types=_MARKETING_CLAIM_TYPES,
                evidence_standards={
                    "attribution": "Must come from a sanctioned attribution model.",
                    "lift": "Requires holdout or quasi-experimental design.",
                    "causal": "Correlation alone is insufficient.",
                },
            )
        )
