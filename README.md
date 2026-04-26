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
without passing evaluation thresholds and (optionally) human approval.

## Known limitations

- The LLM adapters are stubs. The MVP runs entirely on rule-based
  decomposition, classification, and template draft generation. Plug a real
  LLM adapter in behind `infrastructure/llm/base.py::LLM` to gain quality at
  the cost of needing to verify everything it produces.
- Causal, predictive, optimization, and theorem-proving tools are stubs that
  honestly report "no evidence" rather than fabricate one.
- The local document retriever is keyword-based.
- The contradiction checker uses a small antonym table; replace with an NLI
  model or structured policy semantics for production.
- The self-developing module pipeline is currently a documented design plus
  the `/modules/propose` endpoint, which only emits `DRAFT` specs.
