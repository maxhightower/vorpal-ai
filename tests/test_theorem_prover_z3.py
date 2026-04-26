"""Z3LogicalProverTool tests.

Covers:
 - Direct SMT-LIB input: SAT, UNSAT, parse error.
 - Structured form: check-sat, entails (both directions), identifier
   validation, sort validation.
 - Pipeline integration: a LOGICAL claim with a structured payload
   produces a FORMALLY_PROVEN verdict.
"""

from __future__ import annotations

import pytest

from factuality_harness.application.pipeline import (
    FactualityPipeline,
    PipelineRequest,
)
from factuality_harness.domain.claims import Claim
from factuality_harness.domain.epistemic_types import EpistemicType
from factuality_harness.domain.evidence import SourceType, SupportStatus
from factuality_harness.domain.verdicts import Verdict
from factuality_harness.infrastructure.storage.repository import (
    InMemoryAuditRepository,
)
from factuality_harness.infrastructure.tools.base import ToolRequest
from factuality_harness.infrastructure.tools.theorem_prover_z3 import (
    Z3LogicalProverTool,
)


def _claim() -> Claim:
    return Claim(
        text="Does the conclusion follow?",
        parent_question="x",
        epistemic_type=EpistemicType.LOGICAL,
    )


# ---------------------------------------------------------------------------
# Direct SMT-LIB input
# ---------------------------------------------------------------------------


def test_direct_smt_lib_sat_returns_supports():
    tool = Z3LogicalProverTool()
    smt = """
        (declare-const x Int)
        (assert (> x 10))
    """
    result = tool.run(ToolRequest(claim=_claim(), context={"smt_lib": smt}))
    assert result.succeeded
    ev = result.evidence[0]
    assert ev.source_type == SourceType.THEOREM_PROVER
    assert ev.supports_claim == SupportStatus.SUPPORTS
    assert ev.normalized_result["z3_result"] == "sat"
    assert ev.normalized_result["model"] is not None


def test_direct_smt_lib_unsat_returns_contradicts():
    tool = Z3LogicalProverTool()
    smt = """
        (declare-const x Int)
        (assert (> x 10))
        (assert (< x 5))
    """
    result = tool.run(ToolRequest(claim=_claim(), context={"smt_lib": smt}))
    assert result.succeeded
    assert result.evidence[0].supports_claim == SupportStatus.CONTRADICTS
    assert result.evidence[0].normalized_result["z3_result"] == "unsat"


def test_direct_smt_lib_parse_error_fails_cleanly():
    tool = Z3LogicalProverTool()
    result = tool.run(
        ToolRequest(claim=_claim(), context={"smt_lib": "(this is not valid smt"})
    )
    assert not result.succeeded
    assert "SMT-LIB parse error" in (result.error or "")


# ---------------------------------------------------------------------------
# Structured form
# ---------------------------------------------------------------------------


def test_structured_check_sat_satisfiable():
    tool = Z3LogicalProverTool()
    spec = {
        "declares": [{"name": "x", "sort": "Int"}],
        "asserts": ["(> x 0)", "(< x 10)"],
        "query": "check-sat",
    }
    result = tool.run(ToolRequest(claim=_claim(), context={"logic": spec}))
    assert result.succeeded
    assert result.evidence[0].supports_claim == SupportStatus.SUPPORTS


def test_structured_entails_holds():
    """If x > 10 and x < 20, does x > 5? Yes — entailment holds (UNSAT
    of the negation -> SUPPORTS)."""
    tool = Z3LogicalProverTool()
    spec = {
        "declares": [{"name": "x", "sort": "Int"}],
        "asserts": ["(> x 10)", "(< x 20)"],
        "query": "entails",
        "goal": "(> x 5)",
    }
    result = tool.run(ToolRequest(claim=_claim(), context={"logic": spec}))
    assert result.succeeded
    ev = result.evidence[0]
    assert ev.supports_claim == SupportStatus.SUPPORTS
    assert ev.normalized_result["outcome"] == "entailed"


def test_structured_entails_fails_with_counterexample():
    """If x > 10, does x > 20? No — counterexample exists (e.g. x = 11)
    so the entailment is CONTRADICTed and we get a witness back."""
    tool = Z3LogicalProverTool()
    spec = {
        "declares": [{"name": "x", "sort": "Int"}],
        "asserts": ["(> x 10)"],
        "query": "entails",
        "goal": "(> x 20)",
    }
    result = tool.run(ToolRequest(claim=_claim(), context={"logic": spec}))
    assert result.succeeded
    ev = result.evidence[0]
    assert ev.supports_claim == SupportStatus.CONTRADICTS
    assert ev.normalized_result["outcome"] == "counterexample"
    assert ev.normalized_result["counterexample"] is not None
    assert "x" in ev.normalized_result["counterexample"]


def test_structured_rejects_invalid_identifier():
    tool = Z3LogicalProverTool()
    spec = {
        "declares": [{"name": "bad name; (drop)", "sort": "Int"}],
        "asserts": [],
        "query": "check-sat",
    }
    result = tool.run(ToolRequest(claim=_claim(), context={"logic": spec}))
    assert not result.succeeded
    assert "Invalid identifier" in (result.error or "")


def test_structured_rejects_unknown_sort():
    tool = Z3LogicalProverTool()
    spec = {
        "declares": [{"name": "x", "sort": "WeirdSort"}],
        "asserts": [],
        "query": "check-sat",
    }
    result = tool.run(ToolRequest(claim=_claim(), context={"logic": spec}))
    assert not result.succeeded
    assert "WeirdSort" in (result.error or "")


def test_structured_entails_requires_goal():
    tool = Z3LogicalProverTool()
    spec = {
        "declares": [{"name": "x", "sort": "Int"}],
        "asserts": [],
        "query": "entails",
        # no goal
    }
    result = tool.run(ToolRequest(claim=_claim(), context={"logic": spec}))
    assert not result.succeeded
    assert "goal" in (result.error or "").lower()


def test_no_payload_fails_cleanly():
    tool = Z3LogicalProverTool()
    result = tool.run(ToolRequest(claim=_claim(), context={}))
    assert not result.succeeded
    assert "smt_lib" in (result.error or "")


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_logical_claim_with_structured_payload_yields_formally_proven():
    """End-to-end: a LOGICAL claim with a valid SMT-LIB entailment proof
    should reach the FORMALLY_PROVEN verdict via the calibrator."""
    pipeline = FactualityPipeline(audit_repo=InMemoryAuditRepository())
    final = pipeline.run(
        PipelineRequest(
            question="Does this conclusion follow from these premises?",
            extra_context={
                "logic": {
                    "declares": [{"name": "x", "sort": "Int"}],
                    "asserts": ["(> x 10)", "(< x 20)"],
                    "query": "entails",
                    "goal": "(> x 5)",
                }
            },
        )
    )
    trace = pipeline.audit_repo.get(final.audit_id)
    assert trace is not None

    # Z3 emitted THEOREM_PROVER evidence with SUPPORTS.
    prover_ev = [
        e for e in trace.retrieved_evidence
        if e.source_type == SourceType.THEOREM_PROVER
        and e.supports_claim == SupportStatus.SUPPORTS
    ]
    assert prover_ev, "Z3 did not produce SUPPORTS evidence"

    # The verdict calibrator marks LOGICAL + theorem_prover SUPPORTS as
    # FORMALLY_PROVEN.
    logical_verdicts = [
        v for v in trace.verdicts
        if any(
            c.id == v.claim_id and c.epistemic_type == EpistemicType.LOGICAL
            for c in trace.decomposed_claims
        )
    ]
    assert logical_verdicts
    assert logical_verdicts[0].verdict == Verdict.FORMALLY_PROVEN


def test_logical_claim_without_payload_still_unclear():
    """No payload supplied -> Z3 returns clean error, no THEOREM_PROVER
    evidence with SUPPORTS, calibrator routes through UNCLEAR (per the
    LOGICAL branch in uncertainty_calibrator)."""
    pipeline = FactualityPipeline(audit_repo=InMemoryAuditRepository())
    final = pipeline.run(
        PipelineRequest(
            question="Does this conclusion follow from these premises?",
        )
    )
    trace = pipeline.audit_repo.get(final.audit_id)
    logical_verdicts = [
        v for v in trace.verdicts
        if any(
            c.id == v.claim_id and c.epistemic_type == EpistemicType.LOGICAL
            for c in trace.decomposed_claims
        )
    ]
    assert logical_verdicts
    assert logical_verdicts[0].verdict == Verdict.UNCLEAR
