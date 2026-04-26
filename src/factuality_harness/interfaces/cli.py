"""Typer-based CLI for the factuality harness.

Subcommands mirror the API:

  fh answer "..."
  fh audit AUDIT_ID
  fh modules list
  fh modules propose --domain "..." --examples examples.json
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


@evals_app.command("run")
def evals_run() -> None:
    """Run the evaluation suite."""
    from ..evals.run_evals import run_all  # local import keeps CLI startup snappy
    summary = run_all()
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()
