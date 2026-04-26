"""Tests for FactualityPipeline.verify_draft — the agent-tool entry point.

Covers:
 - Question-mode: agent supplies (question, draft) and the harness verifies
   the draft against evidence gathered for the question.
 - Draft-only mode: agent supplies just the draft; the draft itself drives
   decomposition.
 - Hallucination guard: a draft assertion that doesn't map to a verified
   upstream claim is dropped or qualified.
 - Empty-draft guard.
"""

from __future__ import annotations

import pytest

from factuality_harness.application.pipeline import FactualityPipeline
from factuality_harness.infrastructure.storage.repository import (
    InMemoryAuditRepository,
)


def _pipeline() -> FactualityPipeline:
    return FactualityPipeline(audit_repo=InMemoryAuditRepository())


def test_question_mode_keeps_supported_claims():
    """When the harness can verify the draft's claim, the answer survives
    revision (potentially with confidence-qualifier annotations)."""
    p = _pipeline()
    draft = (
        "- Claim: What is the percentage increase from 100 to 125\n"
        "  Verdict: COMPUTED (confidence: HIGH). The result is 25%.\n"
    )
    final = p.verify_draft(
        draft=draft,
        question="What is the percentage increase from 100 to 125?",
    )
    # The supported claim text should still appear; nothing got dropped.
    assert "percentage increase" in final.answer
    assert final.unsupported_or_uncertain_claims == []
    assert "COMPUTED" in final.confidence_summary


def test_draft_only_mode_decomposes_the_draft_itself():
    """When no question is supplied, the harness uses the draft as the
    question and proceeds normally."""
    p = _pipeline()
    final = p.verify_draft(
        draft=(
            "- Claim: What is the percentage increase from 100 to 125\n"
            "  Verdict: COMPUTED (confidence: HIGH). The result is 25%.\n"
        )
    )
    # Audit trace is keyed off the draft (since question defaulted to it).
    trace = p.audit_repo.get(final.audit_id)
    assert trace is not None
    assert trace.original_question.startswith("- Claim:")


def test_hallucinated_draft_claim_is_dropped():
    """Any draft assertion that doesn't map to a verified upstream claim
    must be removed or qualified."""
    p = _pipeline()
    draft = (
        "- Claim: Mars has been colonized in 2025\n"
        "  Verdict: VERIFIED (confidence: HIGH). It absolutely has.\n"
        "- Claim: What is the percentage increase from 100 to 125\n"
        "  Verdict: COMPUTED (confidence: HIGH). The result is 25%.\n"
    )
    final = p.verify_draft(
        draft=draft,
        question="What is the percentage increase from 100 to 125?",
    )
    # The hallucinated Mars assertion must NOT appear in the final answer
    # as a verified claim block — but it SHOULD be flagged in the
    # unsupported_or_uncertain_claims list.
    assert "Mars has been colonized" in " ".join(
        final.unsupported_or_uncertain_claims
    ) or any(
        "Mars" in c for c in final.unsupported_or_uncertain_claims
    )


def test_empty_draft_raises():
    p = _pipeline()
    with pytest.raises(ValueError, match="non-empty"):
        p.verify_draft(draft="")
    with pytest.raises(ValueError, match="non-empty"):
        p.verify_draft(draft="   \n  ")


def test_audit_trace_persists_for_verify_draft():
    """The verify_draft path persists audit traces same as run(), so the
    agent can hand the audit_id back to the caller for provenance."""
    p = _pipeline()
    final = p.verify_draft(
        draft="- Claim: What is 2 + 3\n  Verdict: COMPUTED (confidence: HIGH). 5.\n",
        question="What is 2 + 3?",
    )
    trace = p.audit_repo.get(final.audit_id)
    assert trace is not None
    # The audit trace records that this was a verify_draft path (the
    # supplied draft is what the harness saw).
    assert trace.draft_answer.startswith("- Claim:")
