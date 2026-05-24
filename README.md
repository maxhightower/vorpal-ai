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

Plain LLM answers fail on numerical (bad arithmetic), causal (correlation vs.
causation), predictive (asserting unverified forecasts), procedural (inventing
rules), and optimization ("best" without an objective) claims. Decomposing
first, classifying by epistemic type, and routing to the right verifier turns
these failures into *explicit* uncertainty: an unsupported claim is reported as
unsupported instead of being confidently fabricated.

## Install

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run the API

```sh
uvicorn factuality_harness.interfaces.api:app --reload

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

## Documentation

- [Architecture](docs/architecture.md) — layering rules, pipeline stages, and verdict rules.
- [Domain modules](docs/modules.md) — adding modules, the lifecycle, and the self-developing pipeline.
- [Adapters & integrations](docs/adapters.md) — real LLM/tool/data adapters, code-execution backends, opt-in features, and known limitations.
- [Evals & tests](docs/evals.md) — the regression-tracking eval harness and test layout.
- [Agent tool integration](docs/agent-tool-integration.md) — using the harness as a tool inside an outer agent.
- [Data-acquisition roadmap](docs/data-acquisition-roadmap.md) — the plan to close the acquisition gap.
