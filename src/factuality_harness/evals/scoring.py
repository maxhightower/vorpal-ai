"""Scoring functions for the eval harness.

Each metric reflects a property the harness must preserve:

 - unsupported_claim_rate: fraction of final answers that included an
   unsupported claim. Lower is better.
 - classification_accuracy: fraction of claims classified into the expected
   epistemic type.
 - route_accuracy: fraction of claims routed to the expected tool.
 - evidence_support_accuracy: fraction of expected support relationships that
   were correctly detected.
 - overclaim_rate: fraction of final answers that asserted as VERIFIED a claim
   that should have been UNSUPPORTED/UNCLEAR.
"""

from __future__ import annotations

from pydantic import BaseModel


class CaseResult(BaseModel):
    name: str
    classification_correct: bool | None = None
    routing_correct: bool | None = None
    overclaim: bool | None = None
    unsupported_claim_present: bool | None = None
    notes: str | None = None


class EvalSummary(BaseModel):
    total: int
    classification_accuracy: float | None = None
    route_accuracy: float | None = None
    overclaim_rate: float | None = None
    unsupported_claim_rate: float | None = None
    cases: list[CaseResult] = []


def summarize(results: list[CaseResult]) -> EvalSummary:
    total = len(results)
    if total == 0:
        return EvalSummary(total=0)

    def _rate(predicate, default: bool = False) -> float:
        n = sum(1 for r in results if predicate(r))
        return n / total

    return EvalSummary(
        total=total,
        classification_accuracy=_rate(
            lambda r: r.classification_correct is True
        ),
        route_accuracy=_rate(lambda r: r.routing_correct is True),
        overclaim_rate=_rate(lambda r: r.overclaim is True),
        unsupported_claim_rate=_rate(lambda r: r.unsupported_claim_present is True),
        cases=results,
    )
