# Adapters, backends, and integrations

The harness ships with both stubs (the originals) and real adapters for the
highest-leverage extensions:

| Concern | Real implementation | Module |
|---|---|---|
| Anthropic LLM | `AnthropicAdapter` (uses `anthropic` SDK, defaults to `claude-opus-4-7`, optional adaptive thinking + system-prompt prefix caching) | `infrastructure/llm/anthropic_adapter.py` |
| OpenAI LLM | `OpenAIAdapter` (raw httpx; transport-injectable for testing) | `infrastructure/llm/openai_adapter.py` |
| LLM-backed claim decomposition | `LLMClaimDecomposer` (strict JSON output; hard fallback to rule-based on any parse failure) | `application/llm_claim_decomposer.py` |
| Structured-data SQL | `DuckDBSqlExecutor` (in-memory or external connection; identifier validation; per-request `tables=` registration) | `infrastructure/tools/sql_executor.py` |
| Web retrieval | `TavilyWebRetriever` (httpx; transport-injectable) | `infrastructure/retrieval/tavily_web_retriever.py` |
| Code execution | `LocalSubprocessPythonExecutor` (fresh interpreter per call, wall-clock timeout, optional rlimits — **not** a hardened sandbox) | `infrastructure/tools/python_executor.py` |
| Code execution (Anthropic-hosted) | `AnthropicCodeExecutor` (uses Claude's server-side `code_execution_20260120` tool — no second vendor key, generous free tier) | `infrastructure/tools/python_executor_anthropic.py` |
| Code execution (E2B sandbox) | `E2BPythonExecutor` (ephemeral E2B sandbox per call; `e2b_code_interpreter` is lazy-imported) | `infrastructure/tools/python_executor_e2b.py` |
| Code execution (self-hosted) | `SelfHostedPythonExecutorStub` (placeholder behind the same protocol; documents the Docker / gVisor / Firecracker integration path) | `infrastructure/tools/python_executor_self_hosted.py` |
| Causal A/B testing | `ABTestCausalTool` (two-proportion z-test, no scipy) | `infrastructure/tools/ab_test.py` |
| Causal inference (observational) | `CausalInferenceTool` (DoWhy: backdoor / IV / frontdoor; CI + p-value + refutation) | `infrastructure/tools/causal_inference.py` |
| Forecasting | `ForecastTool` (statsforecast AutoARIMA; calibrated prediction intervals; baseline + direction scoring) | `infrastructure/tools/forecast.py` |
| Linear optimization | `LinearOptimizerTool` (scipy.optimize.linprog / HiGHS; supports min/max + <=/==/>= constraints; refuses to run without an explicit objective and constraints) | `infrastructure/tools/optimizer.py` |

LLM-backed claim decomposition is opt-in via `FACTUALITY_HARNESS_LLM_DECOMPOSER=1` plus
either `ANTHROPIC_API_KEY` (preferred) or `OPENAI_API_KEY`. The `interfaces/factory.py`
helper wires the right decomposer at pipeline construction time.

LLM-assisted **tool-input translation** is similarly opt-in via
`FACTUALITY_HARNESS_LLM_TRANSLATOR=1`. When enabled, the pipeline takes a
plain-English question + raw data in `extra_context["data"]` and uses the LLM
to construct properly-shaped `causal_inference`, `forecast`, `sql`, or
`optimizer` payloads. The translator's output is strictly validated; bad
output produces no payload (the affected tool fails cleanly) — never a
fabricated verdict. See `application/tool_input_translator.py`.

## Data acquisition (Phase A)

The harness has a typed data catalog, real connectors behind a uniform
protocol, and a discovery layer that picks relevant sources per request.

| Piece | Module |
|---|---|
| Catalog model | `domain/catalog.py::DataCatalogEntry`, `DataCatalog` |
| `DataSource` protocol | `infrastructure/data_sources/base.py` |
| DuckDB warehouse connector | `infrastructure/data_sources/warehouse_duckdb.py::DuckDBWarehouseSource` (introspects via `information_schema`) |
| SEC EDGAR REST connector | `infrastructure/data_sources/sec_edgar.py::SECEdgarSource` (scaffolded; static introspect, network on `handle().get_company_facts(cik)`) |
| Discovery layer | `application/data_source_router.py::KeywordDataSourceRouter`, `LLMDataSourceRouter` |
| Pipeline integration | Discovery records to `AuditTrace.data_sources_consulted`; warehouse handle dropped into `ctx['sql_connection']` for the SQL executor |
| Vertical: FinanceModule | `modules/finance.py::FinanceModule` (use `FinanceModule(active=True)`); end-to-end demo at `demos/finance_warehouse.py` |

End-to-end finance demo:

```sh
.venv/bin/python demos/finance_warehouse.py
```

The discovery layer picks `financials`, the LLM-assisted translator
writes a SQL payload against the introspected schema, the SQL executor
runs it against the warehouse's connection, and the verdict is
`SUPPORTED · HIGH` with the warehouse row as evidence.

See [`data-acquisition-roadmap.md`](data-acquisition-roadmap.md) for the
longer-term plan.

## Code-execution backend selection

Selectable via `FACTUALITY_HARNESS_PYTHON_EXECUTOR`:

| Value | Backend |
|---|---|
| `local` (default) | `LocalSubprocessPythonExecutor` — fast, no isolation. |
| `anthropic` | `AnthropicCodeExecutor` — uses Claude's hosted code-execution tool. |
| `e2b` | `E2BPythonExecutor` — needs `pip install e2b_code_interpreter` and `E2B_API_KEY`. |
| `self_hosted` | `SelfHostedPythonExecutorStub` — placeholder for your own Docker/gVisor/Firecracker setup. |

## Contradiction detection

LLM-backed **NLI contradiction detection** is opt-in via
`FACTUALITY_HARNESS_LLM_CONTRADICTION=1`. When enabled, same-claim evidence
pairs are classified by an LLM as `CONTRADICT` / `ENTAIL` / `NEUTRAL` instead
of (or in addition to) the antonym-pair table. This catches non-lexical
conflicts — *"Refunds within 30 days"* vs *"Refunds processed quarterly"* —
that the default detector misses. The LLM detector falls back to the lexical
detector on any parse error or exception, so it can never silently hide a
known antonym-pair contradiction. See `application/contradiction_checker.py`.

## Stubs that remain

Intentionally honest about what they cannot do:

- Theorem prover — `Z3LogicalProverTool` is the default for `LOGICAL` claims.
  Accepts SMT-LIB v2 directly via `extra_context["smt_lib"]` or a structured
  payload (`{declares, asserts, query, goal}`) via `extra_context["logic"]`.
  Maps Z3's `unsat` / `sat` results to `SUPPORTS` / `CONTRADICTS` / `INSUFFICIENT`
  per query semantics; the verdict calibrator promotes `SUPPORTS` to
  `FORMALLY_PROVEN`. NL→SMT translation for open-domain claims is research-grade
  and remains the responsibility of the caller (or an opt-in
  `LLMToolInputTranslator`, which now has a `theorem_prover` spec).
- The fallback `CausalModelStub` (still used when no experimental data is supplied).

## Known limitations

- The local document retriever is keyword-based; `TavilyWebRetriever` covers
  web retrieval but is opt-in (set `TAVILY_API_KEY`).
- `LocalSubprocessPythonExecutor` is **not** a security boundary. For
  untrusted code, swap in an E2B/Modal/Daytona adapter behind the same
  `Tool` protocol.
- The contradiction checker uses a small antonym table; replace with an NLI
  model or structured policy semantics for production.
- The self-developing module pipeline is currently a documented design plus
  the `/modules/propose` endpoint, which only emits `DRAFT` specs.
