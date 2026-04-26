from factuality_harness.application.claim_classifier import RuleBasedClaimClassifier
from factuality_harness.application.claim_decomposer import RuleBasedClaimDecomposer
from factuality_harness.domain.epistemic_types import EpistemicType


def _classify_first(question: str) -> EpistemicType:
    claims = RuleBasedClaimDecomposer().decompose(question)
    classifier = RuleBasedClaimClassifier()
    return classifier.classify(claims[0]).epistemic_type


def test_numerical():
    assert _classify_first("What is the percentage increase from 100 to 125?") == EpistemicType.NUMERICAL


def test_causal():
    assert _classify_first("Did the new ad campaign cause sales to increase?") == EpistemicType.CAUSAL


def test_predictive():
    assert _classify_first("Will demand increase next quarter?") == EpistemicType.PREDICTIVE


def test_optimization():
    assert _classify_first("What is the best allocation across these channels?") == EpistemicType.OPTIMIZATION


def test_procedural():
    assert _classify_first("Is this action allowed under the policy?") == EpistemicType.PROCEDURAL


def test_logical():
    assert _classify_first("Does this conclusion follow from these premises?") == EpistemicType.LOGICAL


def test_speculative():
    assert _classify_first("What if interest rates double?") == EpistemicType.SPECULATIVE


def test_interpretive():
    assert _classify_first("Summarize the strategy.") == EpistemicType.INTERPRETIVE


def test_direct_fact():
    assert _classify_first("What is the capital of France?") == EpistemicType.DIRECT_FACT
