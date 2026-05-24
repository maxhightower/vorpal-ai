# Architecture

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
