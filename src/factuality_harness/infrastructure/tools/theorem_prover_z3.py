"""Z3-backed theorem prover.

Replaces ``TheoremProverStub`` for LOGICAL claims when the input is
formalized. The tool accepts:

  - ``request.context["smt_lib"]`` — a raw SMT-LIB v2 string the user
    or an LLM-assisted translator built. Highest precedence.
  - ``request.context["logic"]`` — a structured payload the tool
    assembles into SMT-LIB itself::

        {
            "declares": [{"name": "x", "sort": "Int"}, ...],
            "asserts":  ["(> x 10)", "(< x 20)"],
            "query":    "check-sat" | "entails",
            "goal":     "(> x 5)"        # required iff query == "entails"
        }

Output semantics:

  - ``query=="check-sat"`` (or absent):
      * UNSAT  → ``CONTRADICTS`` — the asserted formulas are jointly
        unsatisfiable.
      * SAT    → ``SUPPORTS`` — the formulas are satisfiable; a witness
        model is included in ``normalized_result``.
      * UNKNOWN → ``INSUFFICIENT``.
  - ``query=="entails"``: the harness asks whether asserts ⊨ goal. We
    check ``asserts ∧ ¬goal``:
      * UNSAT  → ``SUPPORTS`` (entailment holds).
      * SAT    → ``CONTRADICTS`` (counterexample exists; included in
        ``normalized_result.counterexample``).
      * UNKNOWN → ``INSUFFICIENT``.

When ``supports_claim == SUPPORTS`` the verdict calibrator marks
LOGICAL claims as ``FORMALLY_PROVEN`` (per ``uncertainty_calibrator``).

Honest scope:

  - This is the *deterministic* half of LOGICAL verification. Turning
    natural language into SMT-LIB ("NL→formal") is research-grade for
    open-domain claims. Use this tool on claims a domain module or an
    LLM-assisted translator has already formalized. If translation
    fails, the harness falls back to the existing
    ``TheoremProverStub``-style honest "no proof available" path.
"""

from __future__ import annotations

import re
from typing import Any

import z3

from ...domain.evidence import (
    FreshnessStatus,
    SourceQuality,
    SourceType,
    SupportStatus,
)
from .base import Tool, ToolRequest, ToolResult, build_evidence


# Identifiers we'll let into a structured ``logic`` payload. Prevents
# someone from sneaking arbitrary SMT-LIB through the structured form.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise ValueError(
            f"Invalid identifier {name!r}; allowed: letters, digits, underscore."
        )
    return name


_ALLOWED_SORTS = {"Int", "Real", "Bool"}


def _build_smt_lib_from_structured(spec: dict[str, Any]) -> tuple[str, str]:
    """Assemble an SMT-LIB v2 string from a validated structured payload.

    Returns ``(smt_lib_text, query_kind)`` where ``query_kind`` is one of
    ``"check-sat"`` or ``"entails"``.
    """
    declares = spec.get("declares") or []
    asserts = spec.get("asserts") or []
    query = (spec.get("query") or "check-sat").strip().lower()
    goal = spec.get("goal")

    if not isinstance(declares, list) or not isinstance(asserts, list):
        raise ValueError("'declares' and 'asserts' must be lists.")
    if query not in ("check-sat", "entails"):
        raise ValueError(
            f"query must be 'check-sat' or 'entails'; got {query!r}."
        )
    if query == "entails" and not isinstance(goal, str):
        raise ValueError("query=='entails' requires a string 'goal'.")

    lines: list[str] = []
    for d in declares:
        if not isinstance(d, dict):
            raise ValueError("each entry in 'declares' must be an object.")
        name = _validate_identifier(str(d.get("name", "")))
        sort = str(d.get("sort", "Int"))
        if sort not in _ALLOWED_SORTS:
            raise ValueError(
                f"Sort {sort!r} not allowed; choose from {sorted(_ALLOWED_SORTS)}."
            )
        lines.append(f"(declare-const {name} {sort})")

    for a in asserts:
        if not isinstance(a, str) or not a.strip():
            raise ValueError("each entry in 'asserts' must be a non-empty string.")
        lines.append(f"(assert {a})")

    if query == "entails":
        # Check entailment by adding the negated goal: if the result is
        # UNSAT, the original asserts entail the goal.
        lines.append(f"(assert (not {goal}))")

    return "\n".join(lines), query


def _solve(smt: str, *, timeout_ms: int) -> tuple[z3.CheckSatResult, z3.ModelRef | None]:
    s = z3.Solver()
    s.set("timeout", int(timeout_ms))
    try:
        s.add(z3.parse_smt2_string(smt))
    except z3.Z3Exception as e:
        # Re-raise with a clearer prefix; caller maps to ToolResult error.
        raise z3.Z3Exception(f"SMT-LIB parse error: {e}") from e
    result = s.check()
    if result == z3.sat:
        try:
            return result, s.model()
        except z3.Z3Exception:
            return result, None
    return result, None


class Z3LogicalProverTool:
    name = "theorem_prover"

    def __init__(self, *, default_timeout_ms: int = 5_000) -> None:
        self._timeout_ms = default_timeout_ms

    def run(self, request: ToolRequest) -> ToolResult:
        ctx = request.context
        smt_lib_raw = ctx.get("smt_lib")
        structured = ctx.get("logic")
        timeout_ms = int(ctx.get("logic_timeout_ms", self._timeout_ms))

        # Choose payload source. Direct SMT-LIB always wins when both supplied.
        try:
            if isinstance(smt_lib_raw, str) and smt_lib_raw.strip():
                smt = smt_lib_raw
                query = (
                    "entails"
                    if isinstance(ctx.get("logic_query"), str)
                    and ctx["logic_query"].strip().lower() == "entails"
                    else "check-sat"
                )
            elif isinstance(structured, dict):
                smt, query = _build_smt_lib_from_structured(structured)
            else:
                return ToolResult(
                    succeeded=False,
                    error=(
                        "Z3LogicalProverTool requires context['smt_lib'] (SMT-LIB v2 "
                        "string) or context['logic'] (structured form)."
                    ),
                )
        except ValueError as e:
            return ToolResult(succeeded=False, error=str(e))

        try:
            result, model = _solve(smt, timeout_ms=timeout_ms)
        except z3.Z3Exception as e:
            return ToolResult(succeeded=False, error=str(e))

        # Map Z3 result to our SupportStatus.
        if query == "entails":
            if result == z3.unsat:
                support = SupportStatus.SUPPORTS  # entailment holds
                outcome = "entailed"
            elif result == z3.sat:
                support = SupportStatus.CONTRADICTS  # counterexample exists
                outcome = "counterexample"
            else:
                support = SupportStatus.INSUFFICIENT
                outcome = "unknown"
        else:  # check-sat
            if result == z3.sat:
                support = SupportStatus.SUPPORTS
                outcome = "satisfiable"
            elif result == z3.unsat:
                support = SupportStatus.CONTRADICTS
                outcome = "unsatisfiable"
            else:
                support = SupportStatus.INSUFFICIENT
                outcome = "unknown"

        model_dict: dict[str, str] | None = None
        if model is not None:
            model_dict = {str(d.name()): str(model[d]) for d in model.decls()}

        summary_bits = [f"Z3 {query} -> {result}", f"outcome: {outcome}"]
        if model_dict:
            summary_bits.append(f"witness: {model_dict}")

        return ToolResult(
            succeeded=True,
            evidence=[
                build_evidence(
                    claim_id=request.claim.id,
                    source_type=SourceType.THEOREM_PROVER,
                    source_name="z3",
                    quote_or_result="; ".join(summary_bits),
                    normalized_result={
                        "query": query,
                        "z3_result": str(result),
                        "outcome": outcome,
                        "model": model_dict,
                        "counterexample": (
                            model_dict if outcome == "counterexample" else None
                        ),
                        "smt_lib": smt,
                        "timeout_ms": timeout_ms,
                    },
                    supports_claim=support,
                    source_quality=SourceQuality.AUTHORITATIVE,
                    freshness=FreshnessStatus.NOT_TIME_SENSITIVE,
                    notes=(
                        "Formal proof via Z3. Conclusion is conditional on the "
                        "supplied formalization being correct."
                    ),
                )
            ],
        )


_: Tool = Z3LogicalProverTool()
