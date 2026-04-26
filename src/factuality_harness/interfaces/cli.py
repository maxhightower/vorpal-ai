"""Typer-based CLI for the factuality harness.

Subcommands mirror the API:

  fh answer "..."
  fh audit AUDIT_ID
  fh modules list
  fh modules propose --domain "..." --examples examples.json
  fh modules develop --domain "..." --examples examples.json --output proposal.json
  fh modules evaluate-promotion --name X --pass-rate 0.95 --classification 0.95
  fh modules promote --name X --decision decision.json --yes
  fh evals run
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from ..application.pipeline import PipelineRequest
from ..domain.modules import ModuleSpec, ModuleStatus
from ..infrastructure.retrieval.base import Document
from .factory import build_audit_repo, build_module_registry, build_pipeline

app = typer.Typer(help="Factuality Harness CLI")
modules_app = typer.Typer(help="Manage domain modules.")
evals_app = typer.Typer(help="Run evaluations.")
app.add_typer(modules_app, name="modules")
app.add_typer(evals_app, name="evals")


@app.command()
def answer(
    question: str = typer.Argument(..., help="The question to answer."),
    documents: Optional[Path] = typer.Option(
        None,
        "--documents",
        help="Path to a JSON list of documents [{name, text, effective_date?}, ...].",
    ),
    domain_hint: Optional[str] = typer.Option(None, "--domain"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of text."),
) -> None:
    """Answer a question through the factuality pipeline."""
    docs: list[Document] = []
    if documents is not None:
        raw = json.loads(documents.read_text(encoding="utf-8"))
        docs = [Document.model_validate(d) for d in raw]

    pipeline = build_pipeline()
    final = pipeline.run(
        PipelineRequest(question=question, domain_hint=domain_hint, documents=docs)
    )

    if json_output:
        typer.echo(final.model_dump_json(indent=2))
        return

    typer.echo(final.answer)
    typer.echo("")
    typer.echo(f"Confidence summary: {final.confidence_summary}")
    typer.echo(f"Audit ID: {final.audit_id}")
    if final.unsupported_or_uncertain_claims:
        typer.echo("")
        typer.echo("Unsupported / uncertain claims:")
        for c in final.unsupported_or_uncertain_claims:
            typer.echo(f"  - {c}")


@app.command()
def audit(audit_id: str = typer.Argument(...)) -> None:
    """Print the audit trace for a previous run."""
    repo = build_audit_repo()
    trace = repo.get(audit_id)
    if trace is None:
        typer.echo(f"No audit trace for {audit_id}.", err=True)
        raise typer.Exit(code=1)
    typer.echo(trace.to_json())


@modules_app.command("list")
def modules_list() -> None:
    """List registered domain modules and their lifecycle status."""
    registry = build_module_registry()
    for spec in registry.specs():
        typer.echo(
            f"{spec.name:12s} v{spec.version:6s} [{spec.status.value:10s}] "
            f"- {spec.description}"
        )


@modules_app.command("propose")
def modules_propose(
    domain: str = typer.Option(..., "--domain"),
    examples: Optional[Path] = typer.Option(
        None, "--examples", help="JSON list of example queries."
    ),
) -> None:
    """Produce a DRAFT module spec from example queries (stub)."""
    queries: list[str] = []
    if examples is not None:
        queries = json.loads(examples.read_text(encoding="utf-8"))

    spec = ModuleSpec(
        name=domain,
        description=(
            f"Draft module proposal for {domain}. "
            f"Generated from {len(queries)} example queries. "
            "Must pass evaluation thresholds before promotion."
        ),
        version="0.0.1",
        status=ModuleStatus.DRAFT,
    )
    typer.echo(spec.model_dump_json(indent=2))


@modules_app.command("develop")
def modules_develop(
    domain: str = typer.Option(..., "--domain"),
    examples: Path = typer.Option(
        ..., "--examples", help="JSON list of example queries."
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="Write the full ModuleProposal JSON to this path.",
    ),
) -> None:
    """Run lifecycle stages 1-7 (gap → taxonomy → routes → cases) and emit
    a DRAFT ModuleProposal."""
    from ..application.module_lifecycle import ModuleDevelopmentPipeline

    queries = json.loads(examples.read_text(encoding="utf-8"))
    if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
        raise typer.BadParameter("--examples must be a JSON array of strings")

    pipeline = ModuleDevelopmentPipeline()
    proposal = pipeline.develop_from_queries(
        domain_name=domain, example_queries=queries
    )

    if output is not None:
        output.write_text(proposal.model_dump_json(indent=2), encoding="utf-8")

    typer.echo(
        f"Domain {domain!r}: proposed module with "
        f"{len(proposal.taxonomy)} categor(ies), "
        f"{len(proposal.sources)} source(s), "
        f"{len(proposal.benchmark_case_names)} benchmark case(s)."
    )
    typer.echo(f"is_deployable: {proposal.is_deployable}")
    if not proposal.is_deployable:
        missing = [s.source_id for s in proposal.sources if not s.is_existing]
        typer.echo(f"  missing sources: {missing}")
    if output is not None:
        typer.echo(f"Wrote proposal to {output}")
    else:
        typer.echo("Pass --output PATH to persist the full proposal as JSON.")


@modules_app.command("evaluate-promotion")
def modules_evaluate_promotion(
    module_name: str = typer.Option(..., "--name"),
    module_version: str = typer.Option("0.0.1", "--version"),
    pass_rate: float = typer.Option(..., "--pass-rate"),
    classification_accuracy: float = typer.Option(..., "--classification"),
    overclaim_rate: float = typer.Option(0.0, "--overclaim"),
    shadow_runs: int = typer.Option(0, "--shadow-runs"),
    shadow_agreement: float = typer.Option(1.0, "--shadow-agreement"),
) -> None:
    """Score a module's eligibility for promotion to ACTIVE.

    Outputs a PromotionDecision JSON. Eligibility never auto-promotes —
    the decision still requires human approval (see ``modules promote``).
    """
    from ..application.module_lifecycle import ModuleDevelopmentPipeline
    from ..evals.scoring import EvalSummary

    summary = EvalSummary(
        total_cases=0,
        passed_cases=0,
        failed_cases=0,
        case_pass_rate=pass_rate,
        metrics={
            "case_pass_rate": pass_rate,
            "classification_accuracy": classification_accuracy,
            "overclaim_rate": overclaim_rate,
        },
        cases=[],
    )
    pipeline = ModuleDevelopmentPipeline()
    decision = pipeline.evaluate_promotion(
        module_name=module_name,
        module_version=module_version,
        eval_summary=summary,
        shadow_run_count=shadow_runs,
        shadow_agreement_rate=shadow_agreement,
    )
    typer.echo(decision.model_dump_json(indent=2))


@modules_app.command("promote")
def modules_promote(
    module_name: str = typer.Option(..., "--name"),
    decision_path: Path = typer.Option(
        ...,
        "--decision",
        help="Path to a PromotionDecision JSON produced by `evaluate-promotion`.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Required to actually flip the module to ACTIVE. Without it, "
        "the command refuses even when the decision is eligible.",
    ),
) -> None:
    """Flip a SHADOW module to ACTIVE. The single sanctioned write path."""
    from ..application.module_lifecycle import promote_module
    from ..domain.module_lifecycle import PromotionDecision

    decision = PromotionDecision.model_validate_json(
        decision_path.read_text(encoding="utf-8")
    )
    registry = build_module_registry()
    if registry.get(module_name) is None:
        raise typer.BadParameter(
            f"Module {module_name!r} is not registered."
        )

    result = promote_module(
        module_name, registry, decision=decision, human_approval=yes
    )
    typer.echo(result.model_dump_json(indent=2))
    if not result.promoted:
        if not decision.eligible:
            typer.echo("Refused: decision is not eligible.")
        elif not yes:
            typer.echo(
                "Refused: --yes required to confirm human approval. "
                "Promotion not executed."
            )
        raise typer.Exit(code=1)


@evals_app.command("run")
def evals_run(
    baseline: Optional[Path] = typer.Option(
        None,
        "--baseline",
        help="Compare current run against this baseline JSON; non-zero exit on regression.",
    ),
    save_baseline: Optional[Path] = typer.Option(
        None,
        "--save-baseline",
        help="Write the current run's metrics to this path as a new baseline.",
    ),
    tolerance: float = typer.Option(
        0.0,
        "--tolerance",
        help="Allowed metric movement (in absolute units) before flagging a regression.",
    ),
    min_metric: list[str] = typer.Option(
        [],
        "--min",
        help="Absolute floor, e.g. --min classification_accuracy=0.8 (repeatable).",
    ),
    max_metric: list[str] = typer.Option(
        [],
        "--max",
        help="Absolute ceiling, e.g. --max overclaim_rate=0.1 (repeatable).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON only."),
) -> None:
    """Run the evaluation suite, optionally comparing to a baseline."""
    from ..evals.run_evals import run_with_thresholds  # keep CLI import light

    minima = _parse_metric_pairs(min_metric, "--min")
    maxima = _parse_metric_pairs(max_metric, "--max")

    report = run_with_thresholds(
        baseline_path=str(baseline) if baseline else None,
        save_baseline_path=str(save_baseline) if save_baseline else None,
        tolerance=tolerance,
        minima=minima,
        maxima=maxima,
    )

    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        _print_report(report)

    if report.has_violations:
        raise typer.Exit(code=1)


def _parse_metric_pairs(pairs: list[str], flag: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for raw in pairs:
        if "=" not in raw:
            raise typer.BadParameter(
                f"{flag} expects NAME=VALUE pairs (got {raw!r})."
            )
        name, value = raw.split("=", 1)
        try:
            parsed[name.strip()] = float(value)
        except ValueError as e:
            raise typer.BadParameter(
                f"{flag} value for {name!r} is not numeric: {value!r}."
            ) from e
    return parsed


def _print_report(report) -> None:
    summary = report.summary
    typer.echo(
        f"Cases: {summary.passed_cases}/{summary.total_cases} passed "
        f"({summary.case_pass_rate:.1%})"
    )
    typer.echo("Metrics:")
    for name, value in sorted(summary.metrics.items()):
        typer.echo(f"  {name:30s} {value:.4f}")
    failed = [c for c in summary.cases if not c.passed]
    if failed:
        typer.echo("")
        typer.echo("Failed cases:")
        for c in failed:
            head = c.error or f"{len(c.failed_checks)} check(s) failed"
            typer.echo(f"  - {c.name}: {head}")
            for chk in c.failed_checks:
                typer.echo(
                    f"      • {chk.name}  expected={chk.expected!r}  actual={chk.actual!r}"
                )
    if report.threshold_violations:
        typer.echo("")
        typer.echo("Threshold violations:")
        for v in report.threshold_violations:
            typer.echo(f"  - {v.detail}")
    if report.baseline_violations:
        typer.echo("")
        typer.echo("Baseline regressions:")
        for v in report.baseline_violations:
            typer.echo(f"  - {v.detail}")
    if report.saved_baseline_path:
        typer.echo("")
        typer.echo(f"Saved baseline to {report.saved_baseline_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
