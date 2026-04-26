# Factuality Harness

A claim-governed reasoning harness that improves factual reliability of LLM
systems by **decomposing** user questions into atomic claims, **classifying**
the kind of truth each claim requires, **routing** each claim to the correct
tool or evidence source, **verifying** support, and only then drafting an
answer with explicit uncertainty labels.

> The LLM is not the source of truth. The LLM is the interface, decomposer,
> router, summarizer, and explanation layer. Truth comes from retrieval,
> deterministic computation, rule engines, formal verifiers, structured data,
> causal models, optimizers, simulators, or human-approved domain modules.

## Why claim-first factuality

Plain LLM answers fail badly on:

- **Numerical claims** — they get arithmetic wrong without a calculator.
- **Causal claims** — they conflate correlation with causation.
- **Predictive claims** — they assert a forecast as if it were verified.
- **Procedural claims** — they invent rules instead of citing policy text.
- **Optimization claims** — they declare "best" without an objective function.

Decomposing first, classifying by epistemic type, and routing to the right
verifier converts these failure modes into *explicit* uncertainty: an
unsupported claim is reported as unsupported instead of being confidently
fabricated.

## Architecture

```
        +----------------------+
user -> | interfaces (API/CLI) |
        +-----------+----------+
                    |
                    v
        +-----------+----------+
        |     application      |   <-- pipeline orchestration
        |  (decompose, route,  |
        |   evidence, verify)  |
        +-----+----------+-----+
              |          |
              v          v
   +----------+--+   +---+-----------+
   | infrastruct.|   | domain        |
   | (LLM, tools,|   | (claims,      |
   |  retrieval, |   |  evidence,    |
   |  storage)   |   |  verdicts,    |
   +-------------+   |  modules)     |
                     +---------------+

           +-----------------+
           | modules         |
           | (general, math, |
           |  policy, ...)   |
           +-----------------+
```

Layered architecture rules:

| Layer            | May depend on                | May NOT depend on                  |
|------------------|------------------------------|------------------------------------|
| `domain/`        | nothing                      | application, infra, modules, IO    |
| `application/`   | domain, infra protocols      | vendor SDKs                        |
| `infrastructure/`| domain, vendor SDKs          | application                        |
| `modules/`       | domain                       | infra (uses tools through routing) |
| `interfaces/`    | application, infra           | —                                  |
| `evals/`         | application, modules         | —                                  |

## Pipeline

```
receive_question
  -> decompose_question
  -> classify_claims
  -> choose_domain_modules
  -> route_claims (epistemic type -> tool[s])
  -> execute_tasks (calculator, retriever, rule engine, ...)
  -> check_contradictions
  -> build_evidence_table
  -> assign_verdicts (with strict rules per epistemic type)
  -> generate_draft (template, only from supported evidence)
  -> extract_claims_from_draft
  -> verify_draft_claims (cross-check vs. evidence table)
  -> revise_answer (drop or qualify unsupported claims)
  -> persist_audit_trace
  -> return final_answer
```

Verdict rules enforced explicitly in
`application/uncertainty_calibrator.py`:

- Numerical claims need a `COMPUTATION` or authoritative
  `STRUCTURED_DATA` source.
- Causal claims default to `UNSUPPORTED` unless `CAUSAL_MODEL` evidence
  with `SUPPORTS` status exists.
- Predictive claims are never `VERIFIED`.
- Speculative claims are always `SPECULATIVE`.
- Procedural claims need `RULE_ENGINE` or policy `LOCAL_DOCUMENT` text.
- Logical claims are `FORMALLY_PROVEN` only with `THEOREM_PROVER` evidence.

## Install

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run the API

```sh
uvicorn factuality_harness.interfaces.api:app --reload
```

Then:

```sh
curl -X POST http://localhost:8000/answer \
  -H 'content-type: application/json' \
  -d '{"question": "What is the percentage increase from 100 to 125?"}'
```

Other endpoints:

- `GET /audit/{audit_id}` — full audit trace JSON.
- `GET /modules` — registered domain modules and lifecycle status.
- `POST /modules/propose` — produce a DRAFT module spec from example queries.
- `POST /evals/run` — run the eval suite.

### Regression-tracking eval harness

The CLI exposes a baseline-comparison workflow that turns the eval suite into a
CI gate. Metrics are aggregated per run and can be persisted as a baseline,
compared on subsequent runs, and gated with absolute floors / ceilings.

```sh
# Save a baseline once (e.g. on the main branch)
fh evals run --save-baseline evals/baseline.json

# Compare against the baseline (non-zero exit on regression)
fh evals run --baseline evals/baseline.json --tolerance 0.05

# Or apply absolute thresholds without a baseline
fh evals run --min case_pass_rate=0.95 --max overclaim_rate=0.05
```

Headline metrics surfaced per run:

| Metric | Direction | Meaning |
|---|---|---|
| `case_pass_rate` | higher is better | Fraction of cases where every assertion passed. |
| `classification_accuracy` | higher is better | Fraction of cases asserting on claim type that matched. |
| `verdict_correctness_rate` | higher is better | Fraction of cases asserting on verdicts that matched all of them. |
| `overclaim_rate` | lower is better | Fraction of cases that produced a forbidden verdict (e.g. VERIFIED on a PREDICTIVE claim). |
| `hallucination_rate` | lower is better | Fraction of cases that produced a forbidden answer substring. |

## CLI

```sh
fh answer "What is the percentage increase from 100 to 125, and did that increase prove the campaign caused growth?"

fh audit AUDIT_ID
fh modules list
fh modules propose --domain "insurance underwriting" --examples examples.json
fh evals run
```

Pass policy/contextual documents for procedural and contradiction tests:

```sh
echo '[
  {"name":"DocA","text":"Employees may work remotely up to five days per week.","effective_date":"2024-01-01"},
  {"name":"DocB","text":"As of March 2026, employees must work in office three days per week.","effective_date":"2026-03-01"}
]' > policy.json

fh answer "Can employees work remotely full-time?" --documents policy.json
```

## Tests

```sh
pytest
```

Test files map to the seven spec checkpoints:

- `tests/test_claim_decomposition.py`
- `tests/test_claim_classification.py`
- `tests/test_router.py`
- `tests/test_calculator_tool.py`
- `tests/test_evidence_table.py`
- `tests/test_final_verifier.py`
- `tests/test_pipeline_smoke.py`

## Adding a new domain module

1. Subclass `factuality_harness.modules.base.BaseDomainModule`.
2. Define a `ModuleSpec(name=..., status=ModuleStatus.DRAFT, ...)`.
3. Override `applies_to`, `enrich_claims`, `required_evidence_for`, and
   optionally `validate_evidence`.
4. Register the module in `interfaces/factory.py::build_module_registry`.
5. Author benchmark cases under `evals/datasets/`.
6. Run `fh evals run` and only promote to ACTIVE when thresholds pass.

## Module lifecycle

- `DRAFT` — written but not run; never influences final answers.
- `SHADOW` — runs in parallel for evaluation; cannot influence final answers.
- `ACTIVE` — allowed to influence final answers.
- `DEPRECATED` — retained for audit; no longer used.

The system may *propose* modules, but it must not promote them to `ACTIVE`
without passing evaluation thresholds and human approval.

### Self-developing module pipeline

`application/module_lifecycle.py::ModuleDevelopmentPipeline` implements the
spec's 10 stages:

```sh
# Stages 1-7 — gap → taxonomy → routes → cases. Output is a ModuleProposal.
fh modules develop --domain finance --examples finance_queries.json --output proposal.json

# Stage 8 — eval the proposal's benchmark cases through the harness.
# (run `fh evals run` against the proposal's cases via the run_module_evals API.)

# Stage 10 — score the module's eligibility for ACTIVE.
fh modules evaluate-promotion --name finance --pass-rate 0.95 \
    --classification 0.95 --overclaim 0.0 --shadow-runs 50 --shadow-agreement 0.9 \
    > decision.json

# The single sanctioned promotion path. Refuses without --yes.
fh modules promote --name finance --decision decision.json --yes
```

Safety boundaries enforced by the pipeline:

- `ModuleProposal.is_deployable` blocks proposals that depend on tools the
  harness doesn't have wired in — no silent failures from missing connectors.
- `evaluate_promotion` only decides eligibility; it never flips status.
- `promote_module` is the only sanctioned write path. It refuses without
  explicit `human_approval=True` (CLI: `--yes`), even on eligible decisions.
- SHADOW modules' verdicts are recorded to `AuditTrace.shadow_verdicts` for
  offline metric collection but **never** affect `FinalAnswer`. Exceptions
  inside a shadow module are swallowed; they cannot crash the live pipeline.

## Real adapters available

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

### Data acquisition (Phase A)

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

**Code-execution backend** is selectable via `FACTUALITY_HARNESS_PYTHON_EXECUTOR`:

| Value | Backend |
|---|---|
| `local` (default) | `LocalSubprocessPythonExecutor` — fast, no isolation. |
| `anthropic` | `AnthropicCodeExecutor` — uses Claude's hosted code-execution tool. |
| `e2b` | `E2BPythonExecutor` — needs `pip install e2b_code_interpreter` and `E2B_API_KEY`. |
| `self_hosted` | `SelfHostedPythonExecutorStub` — placeholder for your own Docker/gVisor/Firecracker setup. |

LLM-backed **NLI contradiction detection** is opt-in via
`FACTUALITY_HARNESS_LLM_CONTRADICTION=1`. When enabled, same-claim evidence
pairs are classified by an LLM as `CONTRADICT` / `ENTAIL` / `NEUTRAL` instead
of (or in addition to) the antonym-pair table. This catches non-lexical
conflicts — *"Refunds within 30 days"* vs *"Refunds processed quarterly"* —
that the default detector misses. The LLM detector falls back to the lexical
detector on any parse error or exception, so it can never silently hide a
known antonym-pair contradiction. See `application/contradiction_checker.py`.

Stubs that remain (intentionally honest about what they cannot do):

- Theorem prover (Z3/Lean adapters can be added behind the same protocol).
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
