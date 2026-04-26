"""FinanceModule.

The first vertical wired end-to-end through the data-acquisition stack.
Stays DRAFT by default (no influence on final answers); flip to ACTIVE
explicitly when you wire it to a real warehouse + register that warehouse
in the data catalog.

Usage::

    from factuality_harness.modules.finance import FinanceModule
    from factuality_harness.infrastructure.data_sources.warehouse_duckdb import (
        DuckDBWarehouseSource,
    )
    from factuality_harness.domain.catalog import DataCatalog

    src = DuckDBWarehouseSource(
        source_id="financials",
        description="Quarterly revenue per company.",
        keywords=["revenue", "quarterly"],
    )
    src.load_table("quarterly_revenue", [...])

    catalog = DataCatalog()
    catalog.register(src.introspect())

    pipeline = FactualityPipeline(
        module_registry=ModuleRegistry(modules=[FinanceModule(active=True)]),
        data_catalog=catalog,
        data_source_router=KeywordDataSourceRouter(),
    )

The module declares its claim taxonomy + per-claim evidence requirements
so the verdict calibrator and the data-source router both see what
"finance evidence" needs to look like.
"""

from __future__ import annotations

import re

from ..domain.claims import Claim
from ..domain.epistemic_types import EpistemicType
from ..domain.evidence import EvidenceRequirement, SourceQuality, SourceType
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


# Keywords that strongly suggest a finance question. Used by ``applies_to``.
_FINANCE_KEYWORDS = re.compile(
    r"\b(revenue|earnings|income|profit|loss|margin|ebitda|cogs|opex|capex|"
    r"valuation|dcf|multiples|liquidity|cash|debt|leverage|equity|"
    r"book value|market cap|p/?e|eps|dividend|cash flow|"
    r"quarterly|fiscal|q[1-4]|fy\d{2,4}|10-?[kq]|sec|"
    r"asset|liability|balance sheet|income statement)\b",
    re.IGNORECASE,
)


class FinanceModule(BaseDomainModule):
    def __init__(self, *, active: bool = False, version: str = "0.1.0") -> None:
        super().__init__(
            ModuleSpec(
                name="finance",
                description=(
                    "Financial claims grounded in structured data sources "
                    "(warehouses, audited filings, SEC EDGAR)."
                ),
                version=version,
                status=ModuleStatus.ACTIVE if active else ModuleStatus.DRAFT,
                claim_types=_FINANCE_CLAIM_TYPES,
                source_policy={
                    "required": "audited filings or sanctioned warehouse",
                    "must_be_in_catalog": True,
                },
                tool_policy={
                    "NUMERICAL": ["calculator", "sql_executor"],
                    "DIRECT_FACT": ["sql_executor", "local_document_retriever"],
                    "PREDICTIVE": ["forecast"],
                    "CAUSAL": ["causal_inference"],
                },
                evidence_standards={
                    "revenue": (
                        "Must come from STRUCTURED_DATA (warehouse) or "
                        "RULE_ENGINE-style filing reference; AUTHORITATIVE quality."
                    ),
                    "margin": "Computed from revenue + cost-of-goods-sold line items.",
                    "valuation": (
                        "Method must be explicit (DCF, comparable multiples, "
                        "transaction multiples). Not a single-source claim."
                    ),
                    "liquidity": "Cash + equivalents + short-term receivables.",
                    "risk": "Quantitative risk requires variance/exposure data.",
                    "forecast": (
                        "Predictive — requires forecast model + uncertainty "
                        "intervals; never reported as VERIFIED."
                    ),
                },
            )
        )

    # ------------------------------------------------------------------
    # BaseDomainModule overrides
    # ------------------------------------------------------------------

    def applies_to(self, question: str, claims: list[Claim]) -> float:
        text = question + " " + " ".join(c.text for c in claims)
        matches = _FINANCE_KEYWORDS.findall(text)
        if not matches:
            return 0.0
        # Cap at 1.0; one strong match is plenty.
        return min(1.0, len(matches) / 3.0)

    def required_evidence_for(self, claim: Claim) -> list[EvidenceRequirement]:
        etype = claim.epistemic_type

        if etype == EpistemicType.NUMERICAL:
            return [
                EvidenceRequirement(
                    claim_type=etype,
                    required_source_types=[
                        SourceType.STRUCTURED_DATA,
                        SourceType.COMPUTATION,
                    ],
                    minimum_source_quality=SourceQuality.AUTHORITATIVE,
                    requires_computation=True,
                    requires_date_check=True,
                    requires_contradiction_search=True,
                    rationale=(
                        "Finance numerical claims must come from a warehouse "
                        "or audited filing — never from prose."
                    ),
                )
            ]
        if etype == EpistemicType.DIRECT_FACT:
            return [
                EvidenceRequirement(
                    claim_type=etype,
                    required_source_types=[
                        SourceType.STRUCTURED_DATA,
                        SourceType.LOCAL_DOCUMENT,
                    ],
                    minimum_source_quality=SourceQuality.AUTHORITATIVE,
                    requires_date_check=True,
                    rationale=(
                        "Finance direct-fact claims must cite a warehouse "
                        "table or a primary source filing."
                    ),
                )
            ]
        if etype == EpistemicType.PREDICTIVE:
            return [
                EvidenceRequirement(
                    claim_type=etype,
                    required_source_types=[SourceType.FORECAST_MODEL],
                    minimum_source_quality=SourceQuality.SECONDARY,
                    requires_computation=False,
                    requires_human_review=True,
                    rationale=(
                        "Forecasts must report uncertainty intervals; never "
                        "label as VERIFIED. Human review recommended."
                    ),
                )
            ]
        if etype == EpistemicType.CAUSAL:
            return [
                EvidenceRequirement(
                    claim_type=etype,
                    required_source_types=[SourceType.CAUSAL_MODEL],
                    minimum_source_quality=SourceQuality.PRIMARY,
                    requires_human_review=True,
                    rationale=(
                        "Causal finance claims need experimental / "
                        "quasi-experimental evidence; correlation is not enough."
                    ),
                )
            ]
        return []
