"""LLM-assisted tool-input translator.

Several tools — forecasting, causal inference, SQL, optimization — accept
structured payloads in ``request.context``. Hand-writing those payloads is
the biggest piece of friction left in the harness's UX. This translator
takes a natural-language question + whatever raw data the caller passed
in ``extra_context["data"]`` and asks an LLM to fill in the right tool
payloads.

Critically, this is *only* a propose step. The harness's deterministic
tools still run on whatever payload the translator emits and the verdict
calibrator still applies its per-type rules. A bad translation produces
either a clean tool error or a verdict that fails its evidence
requirements — never a fabricated "verified" claim.

The translator is opt-in. With no translator configured, the pipeline
behaves exactly as before.

Usage::

    pipeline = FactualityPipeline(
        tool_input_translator=LLMToolInputTranslator(llm=AnthropicAdapter()),
        ...
    )

The user passes raw data in ``extra_context["data"]``::

    PipelineRequest(
        question="Did the email cause more conversions?",
        extra_context={
            "data": {
                "experiment": [
                    {"received_email": 0, "converted": 0, "loyalty": 0.3},
                    ...
                ]
            },
        },
    )

The translator produces ``extra_context["causal_inference"]`` for the
DoWhy tool to consume.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ..infrastructure.llm.base import LLM, LLMMessage, LLMRequest


# ---------------------------------------------------------------------------
# Tool-input declarations
# ---------------------------------------------------------------------------


class ToolInputSpec(BaseModel):
    """Declarative description of what a tool reads from request.context.

    Used by the translator to construct an LLM prompt and to validate
    that emitted payloads are at least *shaped* like the example.
    """

    tool_name: str
    context_key: str
    description: str
    example_payload: dict[str, Any] = Field(default_factory=dict)
    required_top_level_keys: list[str] = Field(default_factory=list)


# Built-in specs for the tools that read structured payloads from context.
# Ordered roughly by claim-type frequency.
TOOL_INPUT_SPECS: dict[str, ToolInputSpec] = {
    "causal_inference": ToolInputSpec(
        tool_name="causal_inference",
        context_key="causal_inference",
        description=(
            "Backdoor / IV / frontdoor causal effect estimation via DoWhy. "
            "Use when the claim asks whether a treatment caused an outcome "
            "and the data has at least 30 rows with treatment + outcome + "
            "(common_causes for backdoor or instruments for IV)."
        ),
        example_payload={
            "method": "backdoor",
            "data": [{"treatment_col": 1, "outcome_col": 2.5, "X1": 0.4}],
            "treatment": "treatment_col",
            "outcome": "outcome_col",
            "common_causes": ["X1"],
            "confidence_level": 95,
            "refute": True,
        },
        required_top_level_keys=["data", "treatment", "outcome"],
    ),
    "forecast": ToolInputSpec(
        tool_name="forecast",
        context_key="forecast",
        description=(
            "Time-series forecast via AutoARIMA. Use when the claim asks "
            "about a future value or trend and there is a series of "
            "dated observations (>=10 points)."
        ),
        example_payload={
            "series": [
                {"timestamp": "2024-01-01", "value": 100.0},
                {"timestamp": "2024-01-08", "value": 102.0},
            ],
            "horizon": 8,
            "frequency": "W",
            "confidence_level": 95,
            "baseline": 102.0,
            "direction": "up",
        },
        required_top_level_keys=["series", "horizon"],
    ),
    "sql_executor": ToolInputSpec(
        tool_name="sql_executor",
        context_key="sql",
        description=(
            "DuckDB SQL query against tables registered from the data. Use "
            "when the claim asks about an aggregate or a filtered value "
            "from tabular data."
        ),
        example_payload={
            "sql": "SELECT SUM(revenue) AS total FROM sales WHERE quarter = 'Q1'",
            "tables": {
                "sales": [{"quarter": "Q1", "revenue": 100.0}],
            },
        },
        # The sql_executor reads "sql" and "tables" as separate top-level
        # context keys, not a nested "sql_executor" object — see below for
        # how the translator handles this.
        required_top_level_keys=["sql"],
    ),
    "theorem_prover": ToolInputSpec(
        tool_name="theorem_prover",
        context_key="logic",
        description=(
            "Z3 SMT solver for LOGICAL claims. Use ONLY when the claim is "
            "formal enough to encode as a small SMT-LIB problem with declared "
            "variables and explicit assertions. Never invent variables or "
            "constraints; if the claim is open-domain prose, omit this tool."
        ),
        example_payload={
            "declares": [{"name": "x", "sort": "Int"}],
            "asserts": ["(> x 10)", "(< x 20)"],
            "query": "entails",
            "goal": "(> x 5)",
        },
        required_top_level_keys=["declares", "asserts", "query"],
    ),
    "optimizer": ToolInputSpec(
        tool_name="optimizer",
        context_key="optimization",
        description=(
            "Linear-program optimizer (scipy/HiGHS). Use ONLY when the claim "
            "supplies an explicit objective and explicit constraints — never "
            "to invent an objective from prose."
        ),
        example_payload={
            "kind": "linear",
            "objective": {"sense": "maximize", "coefficients": [3, 5]},
            "variables": [
                {"name": "x", "lower": 0},
                {"name": "y", "lower": 0},
            ],
            "constraints": [
                {"coefficients": [1, 2], "sense": "<=", "rhs": 8},
                {"coefficients": [2, 1], "sense": "<=", "rhs": 8},
            ],
        },
        required_top_level_keys=["kind", "objective", "variables"],
    ),
}


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class ToolInputTranslator(Protocol):
    def translate(
        self,
        *,
        question: str,
        tool_specs: list[ToolInputSpec],
        existing_context: dict[str, Any],
    ) -> dict[str, Any]: ...


class NullToolInputTranslator:
    """No-op translator. Default for the pipeline so behavior is unchanged
    unless an LLM-backed translator is explicitly configured."""

    def translate(
        self,
        *,
        question: str,
        tool_specs: list[ToolInputSpec],
        existing_context: dict[str, Any],
    ) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()

    # Fast path: the whole response is valid JSON. Handles deeply nested
    # payloads that the brace-matching fallback can't.
    if text.startswith("{"):
        try:
            out = json.loads(text)
            return out if isinstance(out, dict) else None
        except json.JSONDecodeError:
            pass

    # Fenced block.
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            out = json.loads(fenced.group(1))
            return out if isinstance(out, dict) else None
        except json.JSONDecodeError:
            pass

    # Last resort: balanced-brace scan from the first ``{``.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    out = json.loads(text[start : i + 1])
                    return out if isinstance(out, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _summarize_data(data: Any, max_sample_rows: int = 3) -> dict[str, Any]:
    """Produce a compact summary of ``extra_context['data']`` for the prompt.

    Tabular shapes (lists of dicts) are described by columns + dtype hints +
    a small sample. Other shapes are described by Python type only.
    """
    if isinstance(data, dict):
        return {k: _summarize_data(v, max_sample_rows) for k, v in data.items()}
    if isinstance(data, list) and data and isinstance(data[0], dict):
        columns = list(data[0].keys())
        dtypes = {k: type(data[0][k]).__name__ for k in columns}
        return {
            "shape": "table",
            "columns": columns,
            "dtypes": dtypes,
            "row_count": len(data),
            "sample": data[:max_sample_rows],
        }
    if isinstance(data, list):
        return {
            "shape": "list",
            "length": len(data),
            "sample": data[:max_sample_rows],
        }
    return {"type": type(data).__name__, "preview": str(data)[:200]}


_SYSTEM_PROMPT = """\
You translate user questions into structured tool payloads.

Rules:
1. Output ONLY a JSON object whose keys are tool names. Each value is the
   payload that the named tool will receive in request.context.
2. Only include tools whose payload you can fully populate from the supplied
   data. If you can't, OMIT that tool entirely.
3. NEVER invent column names. NEVER invent data points. Use only the column
   names that appear in the supplied data summary.
4. NEVER invent an objective function or constraints for the optimizer. The
   optimizer is only included if the question explicitly specifies them.
5. Output JSON only, no prose, no code fences.
"""


def _build_user_prompt(
    question: str,
    tool_specs: list[ToolInputSpec],
    existing_context: dict[str, Any],
) -> str:
    data_summary = _summarize_data(existing_context.get("data"))
    tool_descriptions: list[dict[str, Any]] = []
    for spec in tool_specs:
        tool_descriptions.append(
            {
                "tool_name": spec.tool_name,
                "description": spec.description,
                "required_keys": spec.required_top_level_keys,
                "example_payload": spec.example_payload,
            }
        )
    return json.dumps(
        {
            "question": question,
            "available_data_summary": data_summary,
            "tools": tool_descriptions,
        },
        default=str,
        indent=2,
    )


def _validate_payload(payload: Any, spec: ToolInputSpec) -> bool:
    if not isinstance(payload, dict):
        return False
    for key in spec.required_top_level_keys:
        if key not in payload:
            return False
    return True


# ---------------------------------------------------------------------------
# LLM-backed translator
# ---------------------------------------------------------------------------


class LLMToolInputTranslator:
    """Use an LLM to fill in tool payloads from claim + data.

    Falls back to producing nothing on any parse / validation / runtime
    failure. The downstream tools and verdict calibrator handle the
    consequences (a missing payload yields a clean tool error and an
    UNCLEAR verdict; the harness never asserts an unverified result).
    """

    def __init__(
        self,
        *,
        llm: LLM,
        max_tokens: int = 4096,
    ) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    def translate(
        self,
        *,
        question: str,
        tool_specs: list[ToolInputSpec],
        existing_context: dict[str, Any],
    ) -> dict[str, Any]:
        if not tool_specs:
            return {}

        # Skip tools the user already supplied — never overwrite explicit input.
        candidate_specs = [
            s
            for s in tool_specs
            if s.context_key not in existing_context
            and s.tool_name not in existing_context
        ]
        if not candidate_specs:
            return {}

        prompt = _build_user_prompt(question, candidate_specs, existing_context)
        try:
            response = self._llm.complete(
                LLMRequest(
                    messages=[
                        LLMMessage(role="system", content=_SYSTEM_PROMPT),
                        LLMMessage(role="user", content=prompt),
                    ],
                    temperature=0.0,
                    max_tokens=self._max_tokens,
                )
            )
        except Exception:
            return {}

        parsed = _extract_json(response.text)
        if not isinstance(parsed, dict):
            return {}

        out: dict[str, Any] = {}
        spec_by_name = {s.tool_name: s for s in candidate_specs}
        for tool_name, payload in parsed.items():
            spec = spec_by_name.get(tool_name)
            if spec is None:
                continue
            if not _validate_payload(payload, spec):
                continue

            # The sql_executor reads "sql" and "tables" as separate top-level
            # context keys, not a nested "sql_executor" object. Special-case
            # that here so other tools can stay clean.
            if spec.tool_name == "sql_executor":
                if "sql" in payload:
                    out["sql"] = payload["sql"]
                if "tables" in payload and isinstance(payload["tables"], dict):
                    out["tables"] = payload["tables"]
            else:
                out[spec.context_key] = payload

        return out
