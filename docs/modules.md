# Domain modules

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

## Self-developing module pipeline

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
