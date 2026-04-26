# Data-acquisition roadmap

The harness today is strong at **verification** — given evidence, it produces
calibrated verdicts with auditable provenance. It is weak at **acquisition** —
finding the data needed to answer an open-ended question. This document
captures the gap and a two-phase plan to close it.

## Why this is a separate concern

Verification and acquisition have different trust models. Verification needs
deterministic tools and bounded computation. Acquisition needs auth, schema
introspection, freshness/quality tagging, rate limiting, and PII boundaries.
Conflating them is how systems end up confidently fabricating "data" when an
API is down or a credential expired. The harness errs on the side of
refusing to answer rather than guessing — but that means it can only verify
claims about evidence the caller can reach.

## Current acquisition surface

| Concern | Status today |
|---|---|
| Public web | `TavilyWebRetriever` (when keyed) — `DIRECT_FACT` claims work. |
| User-supplied documents | `LocalDocumentRetriever` keyword search. |
| Pure computation | Calculator works without external data. |
| Sandboxed code execution | `LocalSubprocessPythonExecutor` — local computation only. |
| Time-series for forecasts | User must hand-supply via `extra_context["data"]`. |
| Experimental rows for causal inference | User must hand-supply. |
| Tabular data for SQL | DuckDB executor works only against in-memory tables passed in. |
| Financial filings (SEC EDGAR, etc.) | None. |
| Public stats (FRED, Census, OECD, etc.) | None. |
| Internal CRM / billing / product analytics | None. |
| Schema introspection on live connections | None. |

The architecture is set up to add acquisition cleanly — `BaseDomainModule` already has
`source_policy` and `tool_policy` fields, the `Tool` and `Retriever` protocols are uniform
across new connectors, and the LLM-assisted translator is 90% of the discovery
logic. What's missing is connectors and a catalog.

## What "yes, the harness can obtain the data" requires

Four pieces, in roughly this order:

1. **A data catalog.** Registry of sources (tables, APIs, files) with
   descriptions, freshness windows, access controls, and a typed schema.
   Conceptually a typed extension of the existing `ToolInputSpec` system.
2. **Per-source connectors.** Each source is a `Tool` adapter behind the
   existing protocol — uniform shape regardless of backend.
3. **A discovery / source-router pass.** Given the question + claim type
   + catalog, pick which sources to query. The LLM-assisted translator
   we built handles 90% of this; it just needs the catalog as input.
4. **Schema introspection.** When a connector lands, automatically
   extract column names, dtypes, sample rows, and freshness — populate
   the catalog from the live source.

## Phase A — close the loop for one vertical (~1 week)

Goal: end-to-end open-ended Q&A works for **one** realistic data shape, with
the same `LLM-proposes / deterministic-tools-verify` trust model the rest of
the harness uses.

Concrete vertical (example — pick whatever maps to your real use case): a
warehouse with a marketing-events table + a customers table.

Deliverables:

- [ ] **Catalog model.** `DataCatalogEntry { source_id, source_kind, schema, freshness_window, access_policy, description, sample_query }`. Lives next to `ToolInputSpec`.
- [ ] **One real connector.** `WarehouseSqlConnector` wrapping SQLAlchemy → DuckDB extension for fan-out (or direct Postgres). Schema introspection populates the catalog automatically.
- [ ] **Discovery pass.** A `DataSourceRouter` that runs after `ClaimClassifier`. For each claim, it looks at the claim type + catalog and proposes which sources to query. The LLM-assisted translator already extended for this: pass the catalog as an additional input.
- [ ] **Catalog → translator hand-off.** When the translator builds `causal_inference` / `forecast` / `sql` payloads, the source rows come from the discovered connector, not from `extra_context["data"]`.
- [ ] **End-to-end test.** A question like *"Did the spring email campaign drive more conversions among loyalty-program members?"* runs against fixture-data warehouses and produces a real DoWhy verdict with no hand-written tool payload.
- [ ] **Trace updates.** `AuditTrace` gets `data_sources_consulted: list[CatalogEntry]` so the audit log records which sources were touched and which were skipped.

Architectural rule: connectors are *only* allowed to **return data and metadata**. They don't classify claims, assign verdicts, or produce evidence directly — those continue to come from the verification tools.

## Phase B — generalize to N verticals (ongoing)

Each new connector is additive: same catalog/translator/router scaffolding, new adapter behind it.

Suggested order, roughly by leverage:

| Connector | Backing source | Effort | Unlock |
|---|---|---|---|
| SEC EDGAR XBRL | `https://www.sec.gov/cgi-bin/browse-edgar` | ~3 days | Public-company financials, real `FinanceModule` |
| FRED | `https://fred.stlouisfed.org/docs/api/fred/` | ~1 day | Macro time-series for forecasting |
| Census / BLS | various REST APIs | ~2 days | Demographic / labor stats |
| Internal data warehouse | Snowflake / BigQuery / Postgres | ~3-5 days per family | Whatever the org actually runs on |
| Internal CRM | Salesforce / HubSpot / etc. | ~3 days each | Customer-level questions |
| Product analytics | Segment / Amplitude / Mixpanel | ~3 days each | Funnel / cohort questions |
| A/B platform | Statsig / Optimizely / LaunchDarkly | ~3 days each | Direct experiment results, complements the existing A/B tool |

Hard prerequisites for Phase B:

- **Eval regression harness must be in place first.** Without it, each connector silently degrades coverage. (This is why I recommended building the eval regression harness as the immediate next step before this work.)
- **Per-source quality / freshness tagging.** Already supported by `Evidence` (`source_quality`, `freshness`); connectors must populate these honestly.
- **Per-source access policy.** Connectors that touch private data need authentication wrappers + audit. The harness's existing audit trail captures the *fact* of access; per-source policy enforcement is the gap.

## Out of scope (deliberately)

- **Vector / semantic retrieval over arbitrary corpora.** Worth doing eventually but its evidence-quality story is messy: chunked-snippet retrieval is easy to overclaim from. If added, treat it as `SECONDARY` quality at best until per-passage citations and provenance are wired in.
- **Real-time streaming sources** (websockets, change-data-capture). Add only if product use cases demand it; otherwise periodic-pull is fine.
- **Bring-your-own-LLM data discovery.** The translator pattern works well for picking sources from a small catalog; it does *not* generalize to "the LLM crawls our infrastructure to find data." That has its own trust model and should be a separate component.

## Reading list (when this work starts)

- `application/tool_input_translator.py` — the existing translator is the core of the discovery layer; the catalog extends it.
- `domain/modules.py::ModuleSpec` — `source_policy` and `tool_policy` fields are the schema for what a domain module is allowed to reach.
- `infrastructure/tools/sql_executor.py` — reference for what a connector looks like behind the `Tool` protocol.
- `evals/run_evals.py` — every connector needs benchmark cases; the eval harness is the place they go.
