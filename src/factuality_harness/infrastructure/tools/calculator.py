"""Deterministic calculator tool.

Recognises a small set of common arithmetic claim shapes (percentage change,
arithmetic expressions, ratios) and emits a computed Evidence record. It does
*not* try to be a general expression evaluator; the goal is high-precision
support for the most common numerical claims, with a clear "I cannot compute
this" fallback so that the verifier does not mark something COMPUTED unless it
truly was.
"""

from __future__ import annotations

import re
from typing import Any

from ...domain.evidence import FreshnessStatus, SourceQuality, SourceType, SupportStatus
from .base import Tool, ToolRequest, ToolResult, build_evidence

# percentage change: "from 100 to 125" / "100 -> 125"
_PCT_CHANGE = re.compile(
    r"(?:from\s+)?(?P<a>-?\d+(?:\.\d+)?)\s*(?:to|->|→)\s*(?P<b>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Plain arithmetic expression of the form "<num> <op> <num>".
_ARITH = re.compile(
    r"(?P<a>-?\d+(?:\.\d+)?)\s*(?P<op>[\+\-\*/x×])\s*(?P<b>-?\d+(?:\.\d+)?)",
)


def _safe_div(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return a / b


class CalculatorTool:
    name = "calculator"

    def run(self, request: ToolRequest) -> ToolResult:
        text = request.claim.text

        # Percentage change has highest priority because it overlaps with arithmetic.
        if "percent" in text.lower() or "%" in text:
            m = _PCT_CHANGE.search(text)
            if m:
                a = float(m.group("a"))
                b = float(m.group("b"))
                pct = _safe_div(b - a, a)
                if pct is None:
                    return ToolResult(
                        succeeded=False,
                        error="Division by zero in percentage change (base value is 0).",
                    )
                pct_value = pct * 100.0
                normalized: dict[str, Any] = {
                    "kind": "percentage_change",
                    "from": a,
                    "to": b,
                    "delta": b - a,
                    "percent_change": pct_value,
                    "formula": "(to - from) / from * 100",
                }
                return ToolResult(
                    evidence=[
                        build_evidence(
                            claim_id=request.claim.id,
                            source_type=SourceType.COMPUTATION,
                            source_name="calculator",
                            quote_or_result=(
                                f"({b} - {a}) / {a} * 100 = {pct_value:.4f}%"
                            ),
                            normalized_result=normalized,
                            supports_claim=SupportStatus.SUPPORTS,
                            source_quality=SourceQuality.AUTHORITATIVE,
                            freshness=FreshnessStatus.NOT_TIME_SENSITIVE,
                        )
                    ]
                )

        m = _ARITH.search(text)
        if m:
            a = float(m.group("a"))
            b = float(m.group("b"))
            op = m.group("op").lower().replace("×", "*").replace("x", "*")
            try:
                if op == "+":
                    result = a + b
                elif op == "-":
                    result = a - b
                elif op == "*":
                    result = a * b
                elif op == "/":
                    if b == 0:
                        return ToolResult(succeeded=False, error="Division by zero.")
                    result = a / b
                else:
                    return ToolResult(succeeded=False, error=f"Unknown operator {op!r}.")
            except (OverflowError, ValueError) as e:
                return ToolResult(succeeded=False, error=f"Arithmetic error: {e}")

            normalized = {
                "kind": "arithmetic",
                "a": a,
                "b": b,
                "op": op,
                "result": result,
            }
            return ToolResult(
                evidence=[
                    build_evidence(
                        claim_id=request.claim.id,
                        source_type=SourceType.COMPUTATION,
                        source_name="calculator",
                        quote_or_result=f"{a} {op} {b} = {result}",
                        normalized_result=normalized,
                        supports_claim=SupportStatus.SUPPORTS,
                        source_quality=SourceQuality.AUTHORITATIVE,
                        freshness=FreshnessStatus.NOT_TIME_SENSITIVE,
                    )
                ]
            )

        return ToolResult(
            succeeded=False,
            error="Calculator could not recognize a computable expression in this claim.",
        )


# Static type compatibility check.
_: Tool = CalculatorTool()
