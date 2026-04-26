"""Eval runner.

Executes each benchmark case through the pipeline and produces a summary. The
runner is deliberately tolerant: it returns soft results and counts so the
harness's behaviour can be tracked over time even before strict pass/fail
thresholds are agreed.
"""

from __future__ import annotations

from typing import Any

from ..application.pipeline import PipelineRequest
from ..infrastructure.retrieval.base import Document
from ..interfaces.factory import build_pipeline
from .datasets import CASES
from .scoring import CaseResult, EvalSummary, summarize


def _expected_first_claim_type(trace_classifications: dict[str, str]) -> str | None:
    if not trace_classifications:
        return None
    return next(iter(trace_classifications.values()))


def run_all() -> dict[str, Any]:
    pipeline = build_pipeline()
    results: list[CaseResult] = []

    for case in CASES:
        documents = [Document.model_validate(d) for d in case.get("documents", [])]
        request = PipelineRequest(
            question=case["question"],
            documents=documents,
        )
        final = pipeline.run(request)
        trace = pipeline.audit_repo.get(final.audit_id)

        expected = case.get("expected", {})
        classification_correct: bool | None = None
        if "first_claim_type" in expected and trace is not None:
            classification_correct = (
                _expected_first_claim_type(trace.claim_classifications)
                == expected["first_claim_type"]
            )

        verdict_values = (
            [v.verdict.value for v in trace.verdicts] if trace is not None else []
        )
        overclaim = None
        if "verdicts_must_not_include" in expected:
            overclaim = any(
                v in verdict_values for v in expected["verdicts_must_not_include"]
            )

        unsupported_present = None
        if "verdicts_must_include" in expected:
            unsupported_present = all(
                v in verdict_values for v in expected["verdicts_must_include"]
            )

        results.append(
            CaseResult(
                name=case["name"],
                classification_correct=classification_correct,
                routing_correct=None,  # routing checks deferred
                overclaim=overclaim,
                unsupported_claim_present=unsupported_present,
                notes=f"final={final.confidence_summary}",
            )
        )

    summary: EvalSummary = summarize(results)
    return summary.model_dump()
