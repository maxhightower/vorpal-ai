from __future__ import annotations

import json

from factuality_harness.application.llm_claim_decomposer import LLMClaimDecomposer
from factuality_harness.domain.epistemic_types import EpistemicType
from factuality_harness.infrastructure.llm.base import LLMRequest
from factuality_harness.infrastructure.llm.mock_llm import MockLLM


def test_uses_llm_output_when_well_formed():
    payload = json.dumps(
        {
            "claims": [
                {
                    "text": "The ad campaign caused the lift",
                    "epistemic_type": "CAUSAL",
                },
                {"text": "The lift was 25%", "epistemic_type": "NUMERICAL"},
            ]
        }
    )
    llm = MockLLM(responder=lambda req: payload)

    decomposer = LLMClaimDecomposer(llm=llm)
    claims = decomposer.decompose("Did the campaign lift sales by 25%?")

    assert [c.epistemic_type for c in claims] == [
        EpistemicType.CAUSAL,
        EpistemicType.NUMERICAL,
    ]
    assert claims[0].text == "The ad campaign caused the lift"


def test_handles_fenced_json_response():
    payload = (
        "Here are the claims:\n"
        "```json\n"
        '{"claims": [{"text": "Capital of France is Paris"}]}\n'
        "```"
    )
    llm = MockLLM(responder=lambda req: payload)
    claims = LLMClaimDecomposer(llm=llm).decompose("What is the capital of France?")
    assert len(claims) == 1
    assert "Paris" in claims[0].text


def test_falls_back_when_llm_returns_garbage():
    llm = MockLLM(responder=lambda req: "not json at all")
    claims = LLMClaimDecomposer(llm=llm).decompose(
        "What is the percentage increase from 100 to 125?"
    )
    # Fallback yields the rule-based decomposition; one claim, no exception.
    assert len(claims) == 1


def test_falls_back_when_llm_raises():
    class _BoomLLM:
        name = "boom"

        def complete(self, request: LLMRequest):
            raise RuntimeError("nope")

    claims = LLMClaimDecomposer(llm=_BoomLLM()).decompose(
        "What is 2 + 2 and is it equal to 4?"
    )
    assert len(claims) >= 1


def test_drops_invalid_entries_but_keeps_valid_ones():
    payload = json.dumps(
        {
            "claims": [
                {"text": "valid claim"},
                {"text": ""},  # empty -> dropped
                {"epistemic_type": "NUMERICAL"},  # missing text -> dropped
                {"text": "another", "epistemic_type": "BOGUS"},  # bad type -> UNKNOWN
            ]
        }
    )
    llm = MockLLM(responder=lambda req: payload)
    claims = LLMClaimDecomposer(llm=llm).decompose("anything?")
    assert [c.text for c in claims] == ["valid claim", "another"]
    assert claims[1].epistemic_type == EpistemicType.UNKNOWN
