"""Integration test: an LLM-backed NLI contradiction detector wired into
the pipeline catches a non-lexical contradiction that the default
lexical detector would miss."""

from __future__ import annotations

from factuality_harness.application.contradiction_checker import (
    LLMContradictionDetector,
)
from factuality_harness.application.pipeline import (
    FactualityPipeline,
    PipelineRequest,
)
from factuality_harness.infrastructure.llm.base import LLMRequest
from factuality_harness.infrastructure.llm.mock_llm import MockLLM
from factuality_harness.infrastructure.retrieval.base import Document
from factuality_harness.infrastructure.storage.repository import (
    InMemoryAuditRepository,
)


# Two policy passages that conflict on substance but share no antonym pairs.
# Both contain "refunds" so the rule engine and document retriever (which use
# exact-token matching) surface both as evidence for the same claim.
DOC_A = Document(
    name="HandbookV1",
    text="Refunds must be issued within 30 days of the request.",
    effective_date="2024-01-01",
)
DOC_B = Document(
    name="HandbookV2",
    text="Refunds are batched and processed once per fiscal quarter.",
    effective_date="2026-01-01",
)


def _nli_responder(req: LLMRequest) -> str:
    """Mock NLI: any pair containing both 'refund' phrases is CONTRADICT."""
    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )
    text = last_user.lower()
    if "30 days" in text and ("quarter" in text or "fiscal" in text):
        return "CONTRADICT"
    return "NEUTRAL"


def test_default_lexical_detector_misses_non_lexical_conflict():
    pipeline = FactualityPipeline(audit_repo=InMemoryAuditRepository())
    final = pipeline.run(
        PipelineRequest(
            question="Are refunds processed within 30 days under the policy?",
            documents=[DOC_A, DOC_B],
        )
    )
    trace = pipeline.audit_repo.get(final.audit_id)
    # Lexical antonym table has no entry for "30 days" vs "quarterly".
    # We should NOT have flagged a contradiction with the default detector.
    assert trace is not None
    assert trace.contradictions_found == []


def test_nli_detector_catches_non_lexical_conflict():
    pipeline = FactualityPipeline(
        audit_repo=InMemoryAuditRepository(),
        contradiction_detector=LLMContradictionDetector(
            llm=MockLLM(responder=_nli_responder)
        ),
    )
    final = pipeline.run(
        PipelineRequest(
            question="Are refunds processed within 30 days under the policy?",
            documents=[DOC_A, DOC_B],
        )
    )
    trace = pipeline.audit_repo.get(final.audit_id)
    assert trace is not None
    assert trace.contradictions_found, (
        "NLI detector should have flagged 30-days vs quarterly as contradicting"
    )
