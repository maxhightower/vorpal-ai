from factuality_harness.application.claim_decomposer import RuleBasedClaimDecomposer


def test_single_question_yields_one_claim():
    d = RuleBasedClaimDecomposer()
    claims = d.decompose("What is the percentage increase from 100 to 125?")
    assert len(claims) == 1
    assert "percentage" in claims[0].text.lower()


def test_compound_question_yields_multiple_claims():
    d = RuleBasedClaimDecomposer()
    claims = d.decompose(
        "What is the percentage increase from 100 to 125, "
        "and did that increase prove the campaign caused growth?"
    )
    assert len(claims) >= 2
    texts = [c.text.lower() for c in claims]
    assert any("percentage" in t for t in texts)
    assert any("campaign" in t for t in texts)


def test_empty_question_returns_no_claims():
    d = RuleBasedClaimDecomposer()
    assert d.decompose("") == []


def test_dedupes_repeated_fragments():
    d = RuleBasedClaimDecomposer()
    claims = d.decompose("It is allowed and it is allowed.")
    assert len(claims) == 1
