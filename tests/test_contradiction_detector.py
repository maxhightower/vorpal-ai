"""Tests for both contradiction detectors and the orchestrator."""

from __future__ import annotations

import pytest

from factuality_harness.application.contradiction_checker import (
    LLMContradictionDetector,
    LexicalContradictionDetector,
    check_contradictions,
)
from factuality_harness.domain.evidence import (
    Evidence,
    SourceType,
    SupportStatus,
)
from factuality_harness.infrastructure.llm.base import LLMRequest
from factuality_harness.infrastructure.llm.mock_llm import MockLLM


# ---------------------------------------------------------------------------
# LexicalContradictionDetector — preserves the existing antonym-table behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("Employees may work remotely", "Employees must work in office"),
        ("Refunds are permitted", "Refunds are prohibited"),
        ("This action is allowed", "This action is forbidden"),
        ("Five days per week", "Three days in the office"),
    ],
)
def test_lexical_detector_catches_antonyms(a, b):
    assert LexicalContradictionDetector().is_contradicting(a, b) is True


@pytest.mark.parametrize(
    "a,b",
    [
        # Non-lexical contradictions — the antonym table can't see them.
        ("Refunds must be issued within 30 days", "Refunds are processed quarterly"),
        ("All meetings start at 9am", "Standups begin at 10am"),
        ("Deploys go through code review", "Direct push to main is the standard process"),
    ],
)
def test_lexical_detector_misses_non_lexical(a, b):
    assert LexicalContradictionDetector().is_contradicting(a, b) is False


def test_lexical_detector_no_false_positive_on_unrelated():
    detector = LexicalContradictionDetector()
    assert (
        detector.is_contradicting(
            "The capital of France is Paris.",
            "The Eiffel Tower opened in 1889.",
        )
        is False
    )


# ---------------------------------------------------------------------------
# LLMContradictionDetector
# ---------------------------------------------------------------------------


def _llm_returning(label: str) -> MockLLM:
    return MockLLM(responder=lambda req: label)


def test_llm_detector_returns_true_on_contradict_label():
    det = LLMContradictionDetector(llm=_llm_returning("CONTRADICT"))
    assert det.is_contradicting("A", "B") is True


def test_llm_detector_returns_false_on_entail_or_neutral():
    assert (
        LLMContradictionDetector(llm=_llm_returning("ENTAIL"))
        .is_contradicting("A", "B")
        is False
    )
    assert (
        LLMContradictionDetector(llm=_llm_returning("NEUTRAL"))
        .is_contradicting("A", "B")
        is False
    )


def test_llm_detector_strips_decoration():
    """Real LLMs sometimes wrap a label in fences, quotes, punctuation."""
    for raw in ['"CONTRADICT"', "  contradict.", "**CONTRADICT**", "- Contradict"]:
        det = LLMContradictionDetector(llm=_llm_returning(raw))
        assert det.is_contradicting("A", "B") is True


def test_llm_detector_catches_non_lexical_contradiction_via_llm():
    """Sanity: when the LLM returns CONTRADICT for a non-lexical pair, the
    detector reports it (the lexical fallback would have missed it)."""
    a = "Refunds must be issued within 30 days."
    b = "Refunds are processed quarterly."
    det = LLMContradictionDetector(llm=_llm_returning("CONTRADICT"))
    assert det.is_contradicting(a, b) is True


def test_llm_detector_falls_back_when_label_unparseable():
    """Garbled output -> use the lexical fallback rather than silently hide
    a contradiction. With a known lexical conflict, fallback should fire."""
    det = LLMContradictionDetector(llm=_llm_returning("totally unrelated text"))
    # Lexical conflict: "permitted" vs "prohibited"
    assert det.is_contradicting("Refunds are permitted.", "Refunds are prohibited.") is True
    # Non-lexical conflict: fallback can't see it, so detector reports False.
    assert (
        det.is_contradicting(
            "Refunds must be issued within 30 days.",
            "Refunds are processed quarterly.",
        )
        is False
    )


def test_llm_detector_falls_back_on_exception():
    class _Boom:
        name = "boom"

        def complete(self, request: LLMRequest):
            raise RuntimeError("nope")

    det = LLMContradictionDetector(llm=_Boom())
    # Lexical fallback still works.
    assert det.is_contradicting("permitted", "prohibited") is True


def test_llm_detector_skips_empty_strings():
    det = LLMContradictionDetector(llm=_llm_returning("CONTRADICT"))
    # Should not call the LLM for empty input; should return False.
    assert det.is_contradicting("", "anything") is False
    assert det.is_contradicting("anything", "   ") is False


# ---------------------------------------------------------------------------
# Orchestrator — recency tie-break + detector injection
# ---------------------------------------------------------------------------


def _ev(claim_id: str, source: str, text: str, effective_date: str | None = None) -> Evidence:
    return Evidence(
        claim_id=claim_id,
        source_type=SourceType.LOCAL_DOCUMENT,
        source_name=source,
        quote_or_result=text,
        normalized_result={"effective_date": effective_date} if effective_date else None,
        supports_claim=SupportStatus.PARTIALLY_SUPPORTS,
    )


def test_orchestrator_uses_injected_detector():
    """Pass a stub detector that flags everything; orchestrator should mark
    one side CONTRADICTS regardless of the lexical content."""

    class _AlwaysContradicts:
        def is_contradicting(self, a, b):
            return True

    a = _ev("c1", "DocA", "alpha")
    b = _ev("c1", "DocB", "beta")
    updated, notes = check_contradictions([a, b], detector=_AlwaysContradicts())
    contradicted = [e for e in updated if e.supports_claim == SupportStatus.CONTRADICTS]
    assert len(contradicted) == 1
    assert notes  # at least one note generated


def test_orchestrator_recency_dominance_picks_older_as_loser():
    a = _ev("c1", "OldDoc", "must work in office", effective_date="2024-01-01")
    b = _ev("c1", "NewDoc", "may work remotely", effective_date="2026-03-01")
    updated, _ = check_contradictions([a, b])
    # Older one (a) should be the loser (CONTRADICTS).
    by_name = {e.source_name: e for e in updated}
    assert by_name["OldDoc"].supports_claim == SupportStatus.CONTRADICTS
    assert by_name["NewDoc"].supports_claim == SupportStatus.PARTIALLY_SUPPORTS


def test_orchestrator_does_not_cross_claims():
    """Disagreements between evidence for different claims are not flagged."""
    a = _ev("claim_1", "DocA", "permitted")
    b = _ev("claim_2", "DocB", "prohibited")
    updated, notes = check_contradictions([a, b])
    assert notes == []
    assert all(
        e.supports_claim == SupportStatus.PARTIALLY_SUPPORTS for e in updated
    )


def test_orchestrator_default_detector_is_lexical():
    """Calling without an explicit detector preserves the legacy behavior."""
    a = _ev("c1", "OldDoc", "five days per week", effective_date="2024-01-01")
    b = _ev("c1", "NewDoc", "three days in office", effective_date="2026-03-01")
    updated, notes = check_contradictions([a, b])
    assert notes  # lexical match hit
