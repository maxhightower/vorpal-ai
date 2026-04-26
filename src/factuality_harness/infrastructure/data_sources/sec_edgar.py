"""SEC EDGAR connector — scaffolded against the real public API.

EDGAR's company-facts JSON endpoint exposes XBRL-tagged financial line
items per CIK. This connector wraps the endpoint behind the ``DataSource``
protocol so the harness can declare it in the catalog and route finance
claims through it.

Network access is required at request time. ``introspect()`` does NOT hit
the network — it returns a static catalog entry describing the source's
shape so the discovery layer can reason about it without making API calls.

Concrete query support is intentionally minimal in the MVP:
``handle().get_company_facts(cik)`` returns the parsed JSON. Higher-level
fact extraction (e.g. "what was Q1 2023 revenue for Apple?") belongs in a
dedicated EDGAR fact tool layered on top.

Honest limits:

 - Rate-limited (10 req/s per SEC fair-access policy).
 - User-agent header required.
 - Tests use ``httpx.MockTransport`` — no real network call.
"""

from __future__ import annotations

from typing import Any

import httpx

from ...domain.catalog import (
    AccessPolicy,
    DataCatalogEntry,
    SourceKind,
    TableSchema,
)


_DEFAULT_BASE_URL = "https://data.sec.gov"


class SECEdgarClient:
    """Thin httpx wrapper. Returned by ``SECEdgarSource.handle()``."""

    def __init__(
        self,
        *,
        user_agent: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            timeout=timeout,
            transport=transport,
            headers={"User-Agent": user_agent},
        )
        self._base_url = base_url.rstrip("/")

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def get_company_facts(self, cik: str | int) -> dict[str, Any]:
        cik_padded = f"{int(cik):010d}"
        response = self._client.get(
            f"{self._base_url}/api/xbrl/companyfacts/CIK{cik_padded}.json"
        )
        response.raise_for_status()
        return response.json()


class SECEdgarSource:
    def __init__(
        self,
        *,
        source_id: str = "sec_edgar",
        user_agent: str = "factuality-harness contact@example.com",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.source_id = source_id
        self._client = SECEdgarClient(user_agent=user_agent, transport=transport)

    def handle(self) -> SECEdgarClient:
        return self._client

    def introspect(self) -> DataCatalogEntry:
        # Static description — no network calls. The "schema" here is the
        # JSON shape every company-facts response shares.
        return DataCatalogEntry(
            source_id=self.source_id,
            kind=SourceKind.REST_API,
            description=(
                "SEC EDGAR company-facts XBRL feed. Public US-issuer "
                "financial line items (revenue, assets, liabilities, "
                "equity, EPS, etc.) reported to the SEC, addressed by CIK."
            ),
            access_policy=AccessPolicy.PUBLIC,
            freshness_window_days=90,
            rest_endpoints=[f"{_DEFAULT_BASE_URL}/api/xbrl/companyfacts/CIK<cik>.json"],
            tables=[
                TableSchema(
                    name="company_facts",
                    description="Per-CIK XBRL fact tree (us-gaap and dei taxonomies).",
                    columns={
                        "cik": "BIGINT",
                        "entityName": "VARCHAR",
                        "facts.us-gaap.<concept>.units.<unit>[]": "object",
                    },
                )
            ],
            sample_query="get_company_facts(cik=320193)  # Apple Inc.",
            keywords=[
                "sec",
                "edgar",
                "filings",
                "xbrl",
                "10-k",
                "10-q",
                "revenue",
                "assets",
                "liabilities",
                "equity",
                "eps",
                "company",
                "ticker",
                "cik",
            ],
        )
