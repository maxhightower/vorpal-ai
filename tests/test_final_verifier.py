from factuality_harness.application.final_verifier import (
    TemplateDraftClaimExtractor,
    revise_answer,
    verify_draft_claims,
)
from factuality_harness.domain.claims import Claim
from factuality_harness.domain.confidence import ConfidenceLevel
from factuality_harness.domain.epistemic_types import EpistemicType
from factuality_harness.domain.evidence import EvidenceTable
from factuality_harness.domain.verdicts import ClaimVerdict, Verdict


def _table_with_one_supported_claim() -> EvidenceTable:
    c = Claim(
        text="The percentage increase from 100 to 125 is 25%",
        parent_question="What is the percentage increase from 100 to 125?",
        epistemic_type=EpistemicType.NUMERICAL,
    )
    v = ClaimVerdict(
        claim_id=c.id,
        verdict=Verdict.COMPUTED,
        confidence=ConfidenceLevel.HIGH,
        rationale="Computed deterministically.",
    )
    return EvidenceTable(
        original_question=c.parent_question,
        claims=[c],
        evidence=[],
        verdicts=[v],
    )


def test_extractor_recovers_claim_from_draft():
    table = _table_with_one_supported_claim()
    draft = (
        f"- Claim: {table.claims[0].text}\n"
        f"  Verdict: COMPUTED (confidence: HIGH).\n"
    )
    claims = TemplateDraftClaimExtractor().extract(draft, table)
    assert len(claims) == 1
    assert claims[0].id == table.claims[0].id


def test_revise_keeps_supported_claim():
    table = _table_with_one_supported_claim()
    draft = (
        f"- Claim: {table.claims[0].text}\n"
        f"  Verdict: COMPUTED (confidence: HIGH). This claim was computed deterministically.\n"
    )
    extractor = TemplateDraftClaimExtractor()
    extracted = extractor.extract(draft, table)
    verdicts = verify_draft_claims(extracted, table)
    final = revise_answer(
        question=table.original_question,
        draft=draft,
        draft_verdicts=verdicts,
        table=table,
        audit_id="audit_test",
    )
    assert "Claim:" in final.answer
    assert final.unsupported_or_uncertain_claims == []


def test_revise_drops_hallucinated_claim():
    table = _table_with_one_supported_claim()
    draft = (
        "- Claim: Mars has been colonized in 2025.\n"
        "  Verdict: VERIFIED (confidence: HIGH). This claim is supported by evidence.\n"
        f"- Claim: {table.claims[0].text}\n"
        f"  Verdict: COMPUTED (confidence: HIGH). This claim was computed deterministically.\n"
    )
    extractor = TemplateDraftClaimExtractor()
    extracted = extractor.extract(draft, table)
    verdicts = verify_draft_claims(extracted, table)
    final = revise_answer(
        question=table.original_question,
        draft=draft,
        draft_verdicts=verdicts,
        table=table,
        audit_id="audit_test",
    )
    assert "Mars has been colonized" not in final.answer
    assert any("Mars" in c for c in final.unsupported_or_uncertain_claims)
