from __future__ import annotations

import json

import pytest

from factuality_harness.application.tool_input_translator import (
    TOOL_INPUT_SPECS,
    LLMToolInputTranslator,
    NullToolInputTranslator,
    _summarize_data,
)
from factuality_harness.infrastructure.llm.base import LLMRequest
from factuality_harness.infrastructure.llm.mock_llm import MockLLM


# ---------------------------------------------------------------------------
# NullToolInputTranslator
# ---------------------------------------------------------------------------


def test_null_translator_returns_empty():
    translator = NullToolInputTranslator()
    out = translator.translate(
        question="anything",
        tool_specs=list(TOOL_INPUT_SPECS.values()),
        existing_context={"data": [1, 2, 3]},
    )
    assert out == {}


# ---------------------------------------------------------------------------
# _summarize_data helper (this is the part the LLM sees)
# ---------------------------------------------------------------------------


def test_summarize_data_describes_table():
    rows = [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "y"},
        {"a": 3, "b": "z"},
        {"a": 4, "b": "w"},
    ]
    summary = _summarize_data(rows)
    assert summary["shape"] == "table"
    assert summary["columns"] == ["a", "b"]
    assert summary["row_count"] == 4
    assert len(summary["sample"]) == 3
    assert summary["dtypes"] == {"a": "int", "b": "str"}


def test_summarize_data_handles_dict_of_tables():
    summary = _summarize_data({
        "experiment": [{"x": 1}, {"x": 2}],
        "lookup": [{"k": "v"}],
    })
    assert summary["experiment"]["shape"] == "table"
    assert summary["lookup"]["shape"] == "table"


def test_summarize_data_handles_scalars():
    out = _summarize_data(42)
    assert out["type"] == "int"


# ---------------------------------------------------------------------------
# LLMToolInputTranslator: valid output
# ---------------------------------------------------------------------------


def _causal_payload_for_data(rows: list[dict]) -> dict:
    return {
        "causal_inference": {
            "method": "backdoor",
            "data": rows,
            "treatment": "received_email",
            "outcome": "converted",
            "common_causes": ["loyalty"],
            "confidence_level": 95,
            "refute": False,
        }
    }


def test_emits_payload_for_causal_question():
    rows = [
        {"received_email": 1, "converted": 1, "loyalty": 0.4},
        {"received_email": 0, "converted": 0, "loyalty": 0.2},
    ]
    payload = _causal_payload_for_data(rows)
    llm = MockLLM(responder=lambda req: json.dumps(payload))

    translator = LLMToolInputTranslator(llm=llm)
    out = translator.translate(
        question="Did the email cause more conversions?",
        tool_specs=[TOOL_INPUT_SPECS["causal_inference"]],
        existing_context={"data": {"experiment": rows}},
    )
    assert "causal_inference" in out
    assert out["causal_inference"]["treatment"] == "received_email"
    assert out["causal_inference"]["outcome"] == "converted"


def test_emits_sql_payload_at_top_level_keys():
    """sql_executor reads ``sql`` and ``tables`` as top-level context keys —
    not nested under ``sql_executor``. The translator must lift them."""
    payload = {
        "sql_executor": {
            "sql": "SELECT SUM(revenue) AS total FROM sales",
            "tables": {"sales": [{"revenue": 100}]},
        }
    }
    llm = MockLLM(responder=lambda req: json.dumps(payload))
    out = LLMToolInputTranslator(llm=llm).translate(
        question="What was total revenue?",
        tool_specs=[TOOL_INPUT_SPECS["sql_executor"]],
        existing_context={},
    )
    assert "sql" in out and "tables" in out
    assert "sql_executor" not in out


def test_handles_fenced_json_response():
    payload = {"forecast": {"series": [{"timestamp": "2024-01-01", "value": 1.0}],
                            "horizon": 4, "frequency": "W"}}
    llm = MockLLM(responder=lambda req: f"```json\n{json.dumps(payload)}\n```")
    out = LLMToolInputTranslator(llm=llm).translate(
        question="will the metric increase?",
        tool_specs=[TOOL_INPUT_SPECS["forecast"]],
        existing_context={},
    )
    assert "forecast" in out


# ---------------------------------------------------------------------------
# LLMToolInputTranslator: graceful failure modes
# ---------------------------------------------------------------------------


def test_falls_back_on_garbage():
    llm = MockLLM(responder=lambda req: "definitely not json")
    out = LLMToolInputTranslator(llm=llm).translate(
        question="anything",
        tool_specs=[TOOL_INPUT_SPECS["forecast"]],
        existing_context={},
    )
    assert out == {}


def test_falls_back_on_llm_exception():
    class _Boom:
        name = "boom"

        def complete(self, request: LLMRequest):
            raise RuntimeError("nope")

    out = LLMToolInputTranslator(llm=_Boom()).translate(
        question="anything",
        tool_specs=[TOOL_INPUT_SPECS["forecast"]],
        existing_context={},
    )
    assert out == {}


def test_drops_payloads_missing_required_keys():
    # forecast requires ``series`` and ``horizon``; this output has neither.
    bad = {"forecast": {"frequency": "W"}}
    llm = MockLLM(responder=lambda req: json.dumps(bad))
    out = LLMToolInputTranslator(llm=llm).translate(
        question="x",
        tool_specs=[TOOL_INPUT_SPECS["forecast"]],
        existing_context={},
    )
    assert out == {}


def test_drops_unknown_tool_keys():
    bad = {"made_up_tool": {"anything": True}}
    llm = MockLLM(responder=lambda req: json.dumps(bad))
    out = LLMToolInputTranslator(llm=llm).translate(
        question="x",
        tool_specs=list(TOOL_INPUT_SPECS.values()),
        existing_context={},
    )
    assert out == {}


def test_does_not_overwrite_user_supplied_context():
    user_payload = {"series": "user-supplied", "horizon": 4}
    llm = MockLLM(
        responder=lambda req: json.dumps(
            {"forecast": {"series": [], "horizon": 8}}
        )
    )
    out = LLMToolInputTranslator(llm=llm).translate(
        question="x",
        tool_specs=[TOOL_INPUT_SPECS["forecast"]],
        # User already set forecast in their context — translator must skip.
        existing_context={"forecast": user_payload},
    )
    assert "forecast" not in out


def test_handles_partial_population():
    """Translator emits forecast but skips causal_inference (which is fine).

    The caller asked for both; the LLM only knows how to fill one.
    """
    payload = {"forecast": {"series": [{"timestamp": "2024-01-01", "value": 1.0}],
                            "horizon": 4}}
    llm = MockLLM(responder=lambda req: json.dumps(payload))
    out = LLMToolInputTranslator(llm=llm).translate(
        question="something",
        tool_specs=[TOOL_INPUT_SPECS["forecast"], TOOL_INPUT_SPECS["causal_inference"]],
        existing_context={},
    )
    assert "forecast" in out
    assert "causal_inference" not in out


def test_skips_when_no_specs_supplied():
    llm = MockLLM(responder=lambda req: "{}")
    out = LLMToolInputTranslator(llm=llm).translate(
        question="x", tool_specs=[], existing_context={}
    )
    assert out == {}
