# Evals and tests

## Regression-tracking eval harness

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
