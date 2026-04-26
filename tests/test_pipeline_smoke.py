"""End-to-end smoke tests covering the spec's required examples."""

from __future__ import annotations

from factuality_harness.application.pipeline import FactualityPipeline, PipelineRequest
from factuality_harness.domain.epistemic_types import EpistemicType
from factuality_harness.domain.verdicts import Verdict
from factuality_harness.infrastructure.retrieval.base import Document
from factuality_harness.infrastructure.storage.repository import InMemoryAuditRepository


def _pipeline() -> FactualityPipeline:
    # Tests don't write JSON files.
    return FactualityPipeline(audit_repo=InMemoryAuditRepository())


def test_numerical_with_causal_compound_question():
    p = _pipeline()
    final = p.run(
        PipelineRequest(
            question=(
                "What is the percentage increase from 100 to 125, "
                "and did that increase prove the campaign caused growth?"
            )
        )
    )
    trace = p.audit_repo.get(final.audit_id)
    assert trace is not None

    types = list(trace.claim_classifications.values())
    assert EpistemicType.NUMERICAL.value in types
    assert EpistemicType.CAUSAL.value in types

    verdicts = [v.verdict for v in trace.verdicts]
    assert Verdict.COMPUTED in verdicts
    # The causal claim must NOT be VERIFIED/SUPPORTED — it has no causal evidence.
    causal_verdicts = [
        v for v in trace.verdicts
        if any(
            c.id == v.claim_id and c.epistemic_type == EpistemicType.CAUSAL
            for c in trace.decomposed_claims
        )
    ]
    assert causal_verdicts
    assert all(v.verdict in (Verdict.UNSUPPORTED, Verdict.UNCLEAR) for v in causal_verdicts)
    assert "25" in final.answer


def test_predictive_question_not_verified():
    p = _pipeline()
    final = p.run(PipelineRequest(question="Will demand increase next quarter?"))
    trace = p.audit_repo.get(final.audit_id)
    assert trace is not None
    assert all(v.verdict != Verdict.VERIFIED for v in trace.verdicts)


def test_procedural_without_policy_is_unclear():
    p = _pipeline()
    final = p.run(PipelineRequest(question="Is this action allowed under the policy?"))
    trace = p.audit_repo.get(final.audit_id)
    assert trace is not None
    assert any(v.verdict in (Verdict.UNCLEAR, Verdict.UNSUPPORTED) for v in trace.verdicts)


def test_remote_work_contradiction_detected():
    p = _pipeline()
    final = p.run(
        PipelineRequest(
            question="Can employees work remotely full-time?",
            documents=[
                Document(
                    name="DocA",
                    text="Employees may work remotely up to five days per week.",
                    effective_date="2024-01-01",
                ),
                Document(
                    name="DocB",
                    text="As of March 2026, employees must work in office three days per week.",
                    effective_date="2026-03-01",
                ),
            ],
        )
    )
    trace = p.audit_repo.get(final.audit_id)
    assert trace is not None
    # The contradiction checker must produce at least one note.
    assert trace.contradictions_found
    # The final answer should not be a flat "yes" — caveats or contradictions surface.
    answer_lower = final.answer.lower()
    assert ("office" in answer_lower) or ("contradict" in answer_lower) or ("qualified" in answer_lower)


def test_audit_trace_persisted_and_serializable():
    p = _pipeline()
    final = p.run(PipelineRequest(question="What is the percentage increase from 50 to 75?"))
    trace = p.audit_repo.get(final.audit_id)
    assert trace is not None
    serialized = trace.to_json()
    assert final.audit_id in serialized
