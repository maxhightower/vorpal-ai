"""FastAPI application.

Endpoints:

 - POST /answer            run the pipeline on a question
 - GET  /audit/{id}        fetch an audit trace
 - GET  /modules           list registered modules + lifecycle status
 - POST /modules/propose   draft a new module spec from example queries (stub)
 - POST /evals/run         run the evaluation suite

All endpoints are thin wrappers over application services.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..application.pipeline import PipelineRequest
from ..domain.modules import ModuleSpec
from ..domain.verdicts import FinalAnswer
from ..domain.audit import AuditTrace
from ..infrastructure.retrieval.base import Document
from .factory import build_audit_repo, build_module_registry, build_pipeline


class AnswerRequestBody(BaseModel):
    question: str
    domain_hint: str | None = None
    documents: list[Document] = []


class ProposeModuleBody(BaseModel):
    domain_name: str
    example_queries: list[str] = []


def create_app() -> FastAPI:
    app = FastAPI(title="Factuality Harness", version="0.1.0")

    # Long-lived singletons. Swap to dependency-injected versions when needed.
    pipeline = build_pipeline()
    audit_repo = pipeline.audit_repo
    registry = pipeline.module_registry

    @app.post("/answer", response_model=FinalAnswer)
    def answer(body: AnswerRequestBody) -> FinalAnswer:
        request = PipelineRequest(
            question=body.question,
            domain_hint=body.domain_hint,
            documents=body.documents,
        )
        return pipeline.run(request)

    @app.get("/audit/{audit_id}", response_model=AuditTrace)
    def audit(audit_id: str) -> AuditTrace:
        trace = audit_repo.get(audit_id)
        if trace is None:
            raise HTTPException(status_code=404, detail=f"audit {audit_id} not found")
        return trace

    @app.get("/modules", response_model=list[ModuleSpec])
    def modules() -> list[ModuleSpec]:
        return registry.specs()

    @app.post("/modules/propose", response_model=ModuleSpec)
    def modules_propose(body: ProposeModuleBody) -> ModuleSpec:
        # Phase 1 of the self-developing module pipeline: produce a DRAFT spec.
        # Subsequent stages (taxonomy, sources, tool routes, validators, evals,
        # shadow deployment, promotion) are not implemented yet and must run
        # before any real module reaches ACTIVE.
        return ModuleSpec(
            name=body.domain_name,
            description=(
                f"Draft module proposal for {body.domain_name}. "
                f"Generated from {len(body.example_queries)} example queries. "
                "Must pass evaluation thresholds before promotion."
            ),
            version="0.0.1",
            status="draft",
        )

    @app.post("/evals/run")
    def evals_run() -> dict:
        # Wire to evals/run_evals.py once cases are stabilized.
        return {"status": "not_implemented", "detail": "Eval harness pending."}

    return app


# `uvicorn factuality_harness.interfaces.api:app` works.
app = create_app()
