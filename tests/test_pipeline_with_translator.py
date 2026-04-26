"""End-to-end: a CAUSAL question + raw confounded data + a mock LLM that
emits a causal_inference payload should produce a real DoWhy verdict.

The mock LLM stands in for a real adapter; the rest of the pipeline runs
unchanged. This test demonstrates that the translator closes the loop
between "user has data, asks a question" and "harness produces a
verified verdict" — without anyone hand-writing a tool payload.
"""

from __future__ import annotations

import json
import warnings

import numpy as np

from factuality_harness.application.pipeline import (
    FactualityPipeline,
    PipelineRequest,
)
from factuality_harness.application.tool_input_translator import (
    LLMToolInputTranslator,
)
from factuality_harness.domain.epistemic_types import EpistemicType
from factuality_harness.domain.verdicts import Verdict
from factuality_harness.infrastructure.llm.mock_llm import MockLLM
from factuality_harness.infrastructure.storage.repository import (
    InMemoryAuditRepository,
)


def _confounded_rows(true_effect: float = 1.5, n: int = 1500, seed: int = 0):
    rng = np.random.RandomState(seed)
    loyalty = rng.normal(0, 1, n)
    received_email = (0.4 * loyalty + rng.normal(0, 1, n) > 0).astype(int)
    converted = (
        true_effect * received_email + 1.2 * loyalty + rng.normal(0, 1, n)
    )
    return [
        {
            "loyalty": float(l),
            "received_email": int(r),
            "converted": float(c),
        }
        for l, r, c in zip(loyalty, received_email, converted)
    ]


def _make_translator_responder(rows):
    """Mock LLM that produces a correct causal_inference payload."""
    payload = {
        "causal_inference": {
            "method": "backdoor",
            "data": rows,
            "treatment": "received_email",
            "outcome": "converted",
            "common_causes": ["loyalty"],
            "confidence_level": 95,
            "refute": False,  # speed up the test
        }
    }
    body = json.dumps(payload)
    return lambda req: body


def test_translator_to_dowhy_recovers_true_effect():
    warnings.filterwarnings("ignore")
    rows = _confounded_rows(true_effect=1.5, n=1500)

    translator = LLMToolInputTranslator(
        llm=MockLLM(responder=_make_translator_responder(rows))
    )
    pipeline = FactualityPipeline(
        audit_repo=InMemoryAuditRepository(),
        tool_input_translator=translator,
    )

    final = pipeline.run(
        PipelineRequest(
            question="Did the email cause more conversions?",
            extra_context={
                # Note: NO causal_inference key here — translator must produce it
                "data": {"experiment": rows},
            },
        )
    )

    trace = pipeline.audit_repo.get(final.audit_id)
    assert trace is not None

    # The translator should have populated extra_context['causal_inference'],
    # which the DoWhy tool then consumes.
    causal_evs = [
        e for e in trace.retrieved_evidence
        if e.source_type.value == "CAUSAL_MODEL"
        and "dowhy" in e.source_name
    ]
    assert causal_evs, "translator did not produce a usable causal_inference payload"

    estimate = causal_evs[0].normalized_result["estimate"]
    assert abs(estimate - 1.5) < 0.2  # recovers the true effect within 0.2

    # And the verdict for the CAUSAL claim should be SUPPORTED.
    causal_verdicts = [
        v for v in trace.verdicts
        if any(
            c.id == v.claim_id and c.epistemic_type == EpistemicType.CAUSAL
            for c in trace.decomposed_claims
        )
    ]
    assert causal_verdicts
    assert causal_verdicts[0].verdict == Verdict.SUPPORTED


def test_translator_does_not_overwrite_user_supplied_payload():
    """If the user already passed a causal_inference payload, the translator
    must NOT overwrite it — even if the LLM would have produced something."""
    rows = _confounded_rows(true_effect=1.5, n=400)
    user_payload = {
        "method": "backdoor",
        "data": rows,
        "treatment": "received_email",
        "outcome": "converted",
        "common_causes": ["loyalty"],
        "refute": False,
    }

    # Mock LLM that would produce a *different* (and wrong) payload.
    bad = json.dumps(
        {
            "causal_inference": {
                "method": "backdoor",
                "data": rows,
                "treatment": "wrong_column",  # would crash DoWhy
                "outcome": "wrong_column",
                "common_causes": [],
                "refute": False,
            }
        }
    )
    translator = LLMToolInputTranslator(llm=MockLLM(responder=lambda req: bad))

    pipeline = FactualityPipeline(
        audit_repo=InMemoryAuditRepository(),
        tool_input_translator=translator,
    )

    warnings.filterwarnings("ignore")
    final = pipeline.run(
        PipelineRequest(
            question="Did the email cause more conversions?",
            extra_context={"causal_inference": user_payload},
        )
    )
    trace = pipeline.audit_repo.get(final.audit_id)

    # The DoWhy tool must have run successfully against the user's payload.
    causal_evs = [
        e for e in trace.retrieved_evidence
        if e.source_type.value == "CAUSAL_MODEL"
        and "dowhy" in e.source_name
    ]
    assert causal_evs, "user payload was overwritten or skipped"
