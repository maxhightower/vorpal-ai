"""FinanceModule (stub).

Declares the kinds of finance claims it would handle but defers actual
verification to future structured-data integrations. Stays in DRAFT so it
cannot influence final answers yet.
"""

from __future__ import annotations

from ..domain.modules import ModuleSpec, ModuleStatus
from .base import BaseDomainModule


_FINANCE_CLAIM_TYPES = [
    "revenue",
    "margin",
    "valuation",
    "liquidity",
    "risk",
    "forecast",
]


class FinanceModule(BaseDomainModule):
    def __init__(self) -> None:
        super().__init__(
            ModuleSpec(
                name="finance",
                description="Financial claims (stub — pending structured-data sources).",
                version="0.0.1",
                status=ModuleStatus.DRAFT,
                claim_types=_FINANCE_CLAIM_TYPES,
                source_policy={"required": "audited filings or sanctioned data sources"},
                evidence_standards={
                    "revenue": "From audited filings only.",
                    "valuation": "Method must be explicit (DCF, multiples, etc.).",
                    "forecast": "Must include scenario assumptions.",
                },
            )
        )
