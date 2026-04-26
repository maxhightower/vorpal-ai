from factuality_harness.application.uncertainty_calibrator import assign_verdicts
from factuality_harness.domain.claims import Claim
from factuality_harness.domain.epistemic_types import EpistemicType
from factuality_harness.domain.evidence import (
    Evidence,
    SourceQuality,
    SourceType,
    SupportStatus,
)
from factuality_harness.domain.verdicts import Verdict


def test_numerical_with_computation_yields_computed():
    c = Claim(text="x", parent_question="x", epistemic_type=EpistemicType.NUMERICAL)
    e = Evidence(
        claim_id=c.id,
        source_type=SourceType.COMPUTATION,
        source_name="calculator",
        quote_or_result="25%",
        supports_claim=SupportStatus.SUPPORTS,
        source_quality=SourceQuality.AUTHORITATIVE,
    )
    v = assign_verdicts([c], [e])[0]
    assert v.verdict == Verdict.COMPUTED


def test_causal_without_causal_evidence_is_unsupported():
    c = Claim(text="x", parent_question="x", epistemic_type=EpistemicType.CAUSAL)
    v = assign_verdicts([c], [])[0]
    assert v.verdict == Verdict.UNSUPPORTED


def test_predictive_never_verified():
    c = Claim(text="x", parent_question="x", epistemic_type=EpistemicType.PREDICTIVE)
    v = assign_verdicts([c], [])[0]
    assert v.verdict in (Verdict.UNCLEAR,)
    assert v.verdict != Verdict.VERIFIED


def test_speculative_yields_speculative():
    c = Claim(text="x", parent_question="x", epistemic_type=EpistemicType.SPECULATIVE)
    v = assign_verdicts([c], [])[0]
    assert v.verdict == Verdict.SPECULATIVE


def test_procedural_with_rule_evidence_supported():
    c = Claim(text="x", parent_question="x", epistemic_type=EpistemicType.PROCEDURAL)
    e = Evidence(
        claim_id=c.id,
        source_type=SourceType.RULE_ENGINE,
        source_name="policy",
        quote_or_result="...",
        supports_claim=SupportStatus.PARTIALLY_SUPPORTS,
        source_quality=SourceQuality.AUTHORITATIVE,
    )
    v = assign_verdicts([c], [e])[0]
    assert v.verdict == Verdict.SUPPORTED
