# Handoff — Factuality Harness session

## What this is

A claim-governed reasoning harness for LLM systems. The load-bearing
principle: **the LLM is not the source of truth — it's the interface,
decomposer, router, summarizer, and explanation layer.** Every factual
claim is decomposed, classified by epistemic type, routed to a
deterministic tool, verified, and only then drafted into an answer.
Unsupported claims are dropped or qualified, never confidently
fabricated.

Branch: `claude/factuality-harness-mvp-B3wes`. 17 commits this session,
245 unit tests passing, the built-in eval suite at 8/8 (100%) on every
metric.

## What this session accomplished

The harness went from "spec on paper" to "every epistemic type has a
real verdict path, plus an agent-tool wrapper, plus a regression-tracked
eval harness." Concretely:

| Commit | What it added |
|---|---|
| `a07a285` | Bootstrap: domain models, claim-first pipeline, rule-based decomposer/classifier, calculator, retrieval stub, FastAPI + CLI, 34 tests, README with architecture. Phases 1–3 of the spec. |
| `f8e6032` | Real two-proportion z-test causal tool (`ABTestCausalTool`); widened calculator regex; sharper UNSUPPORTED rationale that distinguishes "no causal model ran" from "ran, not significant." |
| `40ce7a4` | Real `AnthropicAdapter` (official SDK) + `OpenAIAdapter` (raw httpx) + `LLMClaimDecomposer` (strict JSON, hard fallback). Real `DuckDBSqlExecutor`, `TavilyWebRetriever`, `LocalSubprocessPythonExecutor`. |
| `4e75a49` | Real forecast (`statsforecast` AutoARIMA with calibrated PIs) and observational causal (`dowhy` backdoor / IV / frontdoor with CI + p-value + refutation). |
| `43366ee` | LLM decomposer wiring in factory (opt-in env flag); real LP optimizer (`scipy.optimize.linprog` / HiGHS). |
| `654c193` | `LLMToolInputTranslator` — turns plain-English question + raw data into `causal_inference` / `forecast` / `sql` / `optimizer` payloads with strict validation and hard fallback. |
| `8d96ec5` | **Eval regression harness.** `EvalCase` + `ExpectedOutcome` typed cases, per-assertion `CheckResult`, `EvalSummary` aggregate metrics, baseline JSON persistence + regression detection, `fh evals run` CLI gate (non-zero exit on regression). The first run caught a real `ABTestCausalTool` wiring bug that this commit also fixes. |
| `d39d387` | LLM-backed NLI contradiction detector (opt-in) behind the same `ContradictionDetector` protocol. Catches non-lexical conflicts that the antonym-table heuristic misses. |
| `77cd3ee` | `AnthropicCodeExecutor` + `E2BPythonExecutor` + `SelfHostedPythonExecutorStub`. Backend selectable via `FACTUALITY_HARNESS_PYTHON_EXECUTOR=local|anthropic|e2b|self_hosted`. |
| `0176b7a` + `75a645a` | **Module lifecycle automation** (the spec's 10-stage `ModuleDevelopmentPipeline`). Shadow execution wired into the pipeline (SHADOW modules observe, never affect final answer). `PromotionGate` + `promote_module()` as the single sanctioned write path that respects `requires_human_approval`. |
| `1df4bb8` + `162c501` + `7c996b4` | **Data acquisition Phase A.** Typed catalog, `DataSource` protocol, `DuckDBWarehouseSource` with `information_schema` introspection, `SECEdgarSource` REST scaffold, `KeywordDataSourceRouter` + `LLMDataSourceRouter`, pipeline integration that records consulted sources and drops warehouse handles into `ctx['sql_connection']`. `FinanceModule` rewritten and wired end-to-end against a fixture warehouse. |
| `fd426b7` | `scripts/smoke_test_llm_apis.py` — real-provider verification for the seven LLM-touching components, after the user (correctly) pointed out we'd been overstating "works" for unkeyed mock-tested code. |
| `54060f6` | `pipeline.verify_draft()` — agent tool entry point that fact-checks a pre-written draft. `fh verify` CLI. `docs/agent-tool-integration.md` with tool-use schema, agent-loop snippet, slim payload, and explicit private-infrastructure trust model. |
| `992665a` | **Z3 theorem prover for LOGICAL claims** (closes the last hard stub). Accepts SMT-LIB v2 directly or a structured `{declares, asserts, query, goal}` form. `check-sat` and `entails` queries, with witness models and counterexamples. Verdict calibrator promotes `SUPPORTS` to `FORMALLY_PROVEN`. |

## Why each piece is the way it is

**LLM-touching components are opt-in, deterministic-by-default.** The
claim decomposer, tool-input translator, NLI contradiction detector,
and data-source router all default to rule-based / deterministic
implementations. Their LLM-backed variants activate only when an env
flag plus an API key are both set. This keeps tests deterministic,
keeps cost predictable, and keeps the harness's behavior auditable
when an LLM isn't available.

**Every LLM-backed component has a hard fallback to its deterministic
counterpart.** Garbled output, missing fields, exceptions — all map to
"use the rule-based path." The harness's safety boundary is that an
LLM cannot silently inject an unverified claim. If translation fails,
the relevant tool sees no payload and emits `INSUFFICIENT`; the
calibrator routes the claim through `UNCLEAR` / `UNSUPPORTED`.

**Verdicts are always conservative on the safe side.** `PREDICTIVE` is
never `VERIFIED`. `SPECULATIVE` is always labeled. `CAUSAL` requires
real causal-model evidence — correlation alone gets `UNSUPPORTED`.
`OPTIMIZATION` requires explicit objective + constraints, not implied
ones. `LOGICAL` requires formal proof, not prose argument.

**Module lifecycle is gated, not free.** The 10-stage pipeline can
*propose* a module but cannot promote it to ACTIVE without passing
the `PromotionGate` thresholds AND explicit human approval (default).
`promote_module()` is the single sanctioned write path; every other
caller of `registry.promote(name, ACTIVE)` is a bug.

**Data acquisition is structurally separate from verification.** The
catalog, connectors, and discovery layer are their own concern.
Conflating them with verification is how systems end up confidently
fabricating "data" when an API is down. Connectors only return data
and metadata; they never produce verdicts.

## Capability matrix (current)

| Epistemic type | Backend |
|---|---|
| `NUMERICAL` | calculator + DuckDB SQL |
| `DIRECT_FACT` | local docs + Tavily web (when keyed) |
| `PROCEDURAL` | rule engine over policy text |
| `CAUSAL` (RCT) | A/B z-test |
| `CAUSAL` (observational) | DoWhy backdoor / IV / frontdoor |
| `PREDICTIVE` | AutoARIMA with calibrated prediction intervals |
| `OPTIMIZATION` | scipy/HiGHS LP |
| `LOGICAL` | Z3 SMT solver |
| `INTERPRETIVE` | LLM synthesis with assumption flags |
| `SPECULATIVE` | always SPECULATIVE (correctly conservative) |
| `UNKNOWN` | retrieval fallback |

No hard stubs remain.

## Verified end-to-end vs. wired-but-unverified

This distinction matters and is the single most important thing to
internalize from the session.

### Verified with real computation

- All deterministic tools: calculator, optimizer, theorem prover, SQL
  executor, forecast, causal inference, A/B test, local Python
  executor, rule engine, lexical contradiction.
- Pipeline orchestration: claims flow correctly through every stage.
- The data catalog, DuckDB warehouse introspection, FinanceModule
  applies-to logic, lifecycle gates, eval scoring + baseline.
- The end-to-end finance demo (`demos/finance_warehouse.py`) — the
  question/discovery/SQL/verdict chain is real Python end-to-end (the
  LLM translator is mocked because no key was set).

### Wired but never exercised against a real LLM

- `AnthropicAdapter` / `OpenAIAdapter`
- `AnthropicCodeExecutor` (parser shape was inferred from docs — the
  most fragile piece if the SDK returns different attributes)
- `LLMClaimDecomposer` / `LLMToolInputTranslator` /
  `LLMContradictionDetector` / `LLMDataSourceRouter`

### Wired but never exercised against a real network endpoint

- `TavilyWebRetriever` (mocked `httpx.MockTransport`)
- `SECEdgarSource.handle()` (mocked)
- `E2BPythonExecutor` (fake sandbox factory)

### How to close the gap

Run `scripts/smoke_test_llm_apis.py` with a real key:

```sh
export ANTHROPIC_API_KEY=...   # preferred
python scripts/smoke_test_llm_apis.py
```

Each probe reports `PASS` / `SOFT` / `FAIL` / `SKIP` for one component
and prints the raw response when something doesn't match parser
expectations. Cost: a few cents per full pass. Do this **before**
relying on any of the LLM-backed paths in production.

## How to use the harness

### As a Python library

```python
from factuality_harness.interfaces.factory import build_pipeline
from factuality_harness.application.pipeline import PipelineRequest

pipeline = build_pipeline()
final = pipeline.run(PipelineRequest(question="..."))
# final.answer, final.confidence_summary, final.evidence_table, final.audit_id
```

### As an agent tool (recommended pattern)

```python
final = pipeline.verify_draft(
    draft="...",                    # agent-composed answer
    question="...",                 # optional, drives evidence gathering
    documents=[...],                # optional Document objects
    extra_context={...},            # optional structured context
)
# final.unsupported_or_uncertain_claims  -> what was dropped/qualified
```

See `docs/agent-tool-integration.md` for the full Anthropic tool-use
schema and the recommended slim payload to send back to the agent.

### CLI

```sh
fh answer "..."                                            # full Q&A
fh verify "..." --question "..."                           # fact-check a draft
fh audit AUDIT_ID                                          # full provenance trace
fh modules list / develop / evaluate-promotion / promote   # lifecycle
fh evals run [--save-baseline F] [--baseline F] [--min/--max NAME=VALUE]
```

### FastAPI

```sh
uvicorn factuality_harness.interfaces.api:app --reload
```

`POST /answer`, `GET /audit/{id}`, `GET /modules`, `POST /modules/propose`,
`POST /evals/run`. Private by default — runs on your infrastructure.

### Opt-in env flags (LLM paths)

| Flag | Purpose |
|---|---|
| `FACTUALITY_HARNESS_LLM_DECOMPOSER=1` | Use LLM for claim decomposition. |
| `FACTUALITY_HARNESS_LLM_TRANSLATOR=1` | Use LLM to build tool payloads from raw data. |
| `FACTUALITY_HARNESS_LLM_CONTRADICTION=1` | Use LLM (NLI) for contradiction detection. |
| `FACTUALITY_HARNESS_PYTHON_EXECUTOR=local|anthropic|e2b|self_hosted` | Code-execution backend. |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Required when any of the above are enabled. |
| `TAVILY_API_KEY` | Web retrieval. |
| `E2B_API_KEY` | Required when `FACTUALITY_HARNESS_PYTHON_EXECUTOR=e2b`. |

## How to extend the harness

| To add… | Look at… | Pattern |
|---|---|---|
| A new tool | `infrastructure/tools/calculator.py` | Implement the `Tool` protocol; register in `pipeline.py::default_tools`. |
| A new data connector | `infrastructure/data_sources/warehouse_duckdb.py` | Implement `DataSource` protocol (`introspect()` + `handle()`); register an entry in the catalog. |
| A new LLM-backed variant of an existing component | `application/llm_claim_decomposer.py` | Strict JSON system prompt + hard fallback + opt-in env flag in `interfaces/factory.py`. |
| A new domain module | `modules/finance.py` | Subclass `BaseDomainModule`; override `applies_to`, `required_evidence_for`, optionally `validate_evidence`. |
| A new eval case | `evals/cases.py` | Add an `EvalCase` with structured `ExpectedOutcome` assertions; the runner picks it up automatically. |
| A new connector route | `application/router.py::_DEFAULT_ROUTES` | Add the new tool's name to the relevant `EpistemicType`'s route. |
| A self-developing module | `application/module_lifecycle.py::ModuleDevelopmentPipeline` | Use `develop_from_queries()` to produce a `ModuleProposal`; deploy to SHADOW; collect metrics; gate promotion. |

## Open work / recommended next steps

The remaining work is **additive, not architectural**.

| Item | Effort | Why it matters |
|---|---|---|
| **Run the LLM smoke test with a real key** | minutes | Converts the seven mock-tested LLM components from "wired" to "verified." Necessary before relying on any LLM-backed path in production. |
| **Phase B data connectors** | per-connector, ~1–5 days each | Each new vertical (FRED for macro time series, Postgres/Snowflake/BigQuery for warehouses, Salesforce/HubSpot for CRM) is one file conforming to the `DataSource` protocol + one catalog `register()`. Roadmap in `docs/data-acquisition-roadmap.md`. |
| **Prompt regression evals** | ~3 days | Lock in real-LLM behavior for the seven LLM-backed components so prompt edits don't silently degrade quality. Built on top of `evals/run_evals.py`. |
| **Wire SHADOW metrics aggregation** | ~2 days | The lifecycle pipeline records `ShadowVerdict`s per run but doesn't yet aggregate them into `ShadowMetrics` over time. Promotion currently asks the caller to supply `shadow_run_count` + `shadow_agreement_rate` manually. |
| **Multi-warehouse routing** | ~3 days | The pipeline currently picks the first warehouse that the discovery layer returns. A multi-warehouse query would need either a federated SQL view or per-warehouse query fan-out. |
| **`SECEdgarSource` fact extraction** | ~2 days | The connector is scaffolded; pulling specific XBRL line items (revenue, EPS, etc.) per CIK + period is the next step. |

## Repository tour

```
src/factuality_harness/
  domain/              # Pure business types — Pydantic models, no I/O.
    claims.py, evidence.py, verdicts.py, audit.py, modules.py
    epistemic_types.py, confidence.py
    catalog.py                    # data catalog (Phase A.1)
    module_lifecycle.py           # gap, proposal, promotion gate
  application/         # Orchestration. No vendor SDKs.
    pipeline.py                   # FactualityPipeline.run / verify_draft
    claim_decomposer.py           # rule-based + protocol
    claim_classifier.py           # rule-based regex tables
    router.py                     # epistemic_type -> tool list
    contradiction_checker.py      # lexical + LLM-NLI detectors
    uncertainty_calibrator.py     # the per-type verdict rules
    draft_generator.py            # template draft from evidence
    final_verifier.py             # extract claims, revise, audit
    llm_claim_decomposer.py       # LLM-backed (opt-in)
    tool_input_translator.py      # LLM-backed (opt-in)
    data_source_router.py         # discovery layer
    module_lifecycle.py           # 10-stage development pipeline
    module_registry.py            # DRAFT/SHADOW/ACTIVE/DEPRECATED
  infrastructure/      # Adapters. SDKs and protocols allowed here.
    llm/                          # AnthropicAdapter, OpenAIAdapter, MockLLM
    retrieval/                    # local + Tavily
    tools/                        # calculator, sql, python (4 backends),
                                  # rule engine, theorem prover (Z3),
                                  # ab_test, causal_inference, forecast,
                                  # optimizer
    data_sources/                 # DuckDBWarehouseSource, SECEdgarSource
    storage/                      # JSON audit repository
  modules/             # Domain modules.
    base.py, general.py, math.py, policy.py
    finance.py        # ACTIVE when wired to a warehouse + catalog
    marketing.py, code.py  # DRAFT
  interfaces/          # User-facing entry points.
    api.py            # FastAPI
    cli.py            # Typer (`fh ...`)
    factory.py        # opt-in flag wiring
  evals/               # Regression harness.
    cases.py, scoring.py, baselines.py, run_evals.py

tests/                 # 245 tests
demos/
  spring_promo_ab_test.py        # A/B causal verdict end-to-end
  finance_warehouse.py           # data acquisition + SQL verdict end-to-end
scripts/
  smoke_test_llm_apis.py         # real-provider verification (run with a key)
docs/
  data-acquisition-roadmap.md    # Phase A + B plan
  agent-tool-integration.md      # how to wire verify_draft as an agent tool
  session-handoff.md             # this file
```

## Key files to read first

If you're picking this up cold, in order:

1. `README.md` — high-level architecture + quickstart.
2. `docs/agent-tool-integration.md` — the agent-tool entry point and trust model.
3. `src/factuality_harness/application/pipeline.py` — the orchestrator. Read `run()` and `verify_draft()`.
4. `src/factuality_harness/application/uncertainty_calibrator.py` — the per-type verdict rules. This file encodes the harness's safety boundary.
5. `src/factuality_harness/evals/cases.py` + `run_evals.py` — the regression-tracking eval harness.
6. `docs/data-acquisition-roadmap.md` — the next vertical's plan.

## Test + eval status at handoff

```
pytest -q                    -> 245 passed
fh evals run                 -> Cases: 8/8 passed (100.0%)
                                case_pass_rate              1.0000
                                classification_accuracy     1.0000
                                hallucination_rate          0.0000
                                overclaim_rate              0.0000
                                verdict_correctness_rate    1.0000
```

Every commit on the branch keeps these numbers green. Adding new
features without running `fh evals run` and `pytest -q` first is how
this stops being true.
