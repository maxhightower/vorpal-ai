"""Linear-programming optimizer tool backed by ``scipy.optimize.linprog``.

The harness's policy on optimization claims is firm: there is no "best"
without an explicit objective and explicit constraints. This tool refuses
to run when either is missing — that's the correct behavior, not a bug.

Expected context shape::

    request.context["optimization"] = {
        "kind": "linear",                 # only "linear" supported in MVP
        "objective": {
            "sense": "minimize" | "maximize",
            "coefficients": [c1, c2, ...]   # length = number of variables
        },
        "variables": [
            {"name": "x", "lower": 0, "upper": None},   # None means unbounded
            {"name": "y", "lower": 0, "upper": None},
            ...
        ],
        "constraints": [
            {"coefficients": [a1, a2, ...], "sense": "<=", "rhs": b},
            {"coefficients": [...],          "sense": ">=", "rhs": b},
            {"coefficients": [...],          "sense": "==", "rhs": b},
            ...
        ],
    }

Output: ``OPTIMIZER`` evidence with the solution vector, the objective
value, and the LP status. ``supports_claim=SUPPORTS`` only when the solver
reports an optimal solution (status 0); ``CONTRADICTS`` when the problem
is infeasible/unbounded (no feasible "best" exists, which actively refutes
any "X is optimal" claim).
"""

from __future__ import annotations

from typing import Any

from ...domain.evidence import (
    FreshnessStatus,
    SourceQuality,
    SourceType,
    SupportStatus,
)
from .base import Tool, ToolRequest, ToolResult, build_evidence


_SUPPORTED_SENSES = ("minimize", "maximize")
_SUPPORTED_CONSTRAINT_SENSES = ("<=", ">=", "==")


def _coerce_float_list(values: list[Any], context: str) -> list[float]:
    out: list[float] = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError) as e:
            raise ValueError(f"{context}: non-numeric value {v!r}") from e
    return out


def _bounds_pair(var: dict[str, Any]) -> tuple[float | None, float | None]:
    lb = var.get("lower")
    ub = var.get("upper")
    return (
        float(lb) if lb is not None else None,
        float(ub) if ub is not None else None,
    )


class LinearOptimizerTool:
    name = "optimizer"

    def run(self, request: ToolRequest) -> ToolResult:
        spec = request.context.get("optimization")
        if not isinstance(spec, dict):
            return ToolResult(
                succeeded=False,
                error=(
                    "OptimizerTool requires request.context['optimization'] with "
                    "an explicit objective and constraints. The harness will not "
                    "claim 'optimal' without a formal model."
                ),
            )

        kind = str(spec.get("kind", "linear")).lower()
        if kind != "linear":
            return ToolResult(
                succeeded=False,
                error=f"Only kind='linear' is supported in MVP (got {kind!r}).",
            )

        objective = spec.get("objective")
        variables = spec.get("variables")
        constraints = spec.get("constraints", [])

        if not isinstance(objective, dict) or not isinstance(variables, list) or not variables:
            return ToolResult(
                succeeded=False,
                error="objective (object) and variables (non-empty list) are required.",
            )
        if not isinstance(constraints, list):
            return ToolResult(
                succeeded=False, error="constraints must be a list."
            )

        sense = str(objective.get("sense", "minimize")).strip().lower()
        if sense not in _SUPPORTED_SENSES:
            return ToolResult(
                succeeded=False,
                error=f"objective.sense must be one of {_SUPPORTED_SENSES}; got {sense!r}.",
            )

        n_vars = len(variables)
        var_names: list[str] = []
        try:
            for i, v in enumerate(variables):
                if not isinstance(v, dict):
                    raise ValueError(f"variables[{i}] must be an object")
                var_names.append(str(v.get("name", f"x{i}")))
            bounds = [_bounds_pair(v) for v in variables]

            obj_coeffs = _coerce_float_list(
                list(objective.get("coefficients", [])),
                "objective.coefficients",
            )
            if len(obj_coeffs) != n_vars:
                return ToolResult(
                    succeeded=False,
                    error=(
                        f"objective.coefficients has length {len(obj_coeffs)}; "
                        f"expected {n_vars} (matches variables)."
                    ),
                )

            A_ub: list[list[float]] = []
            b_ub: list[float] = []
            A_eq: list[list[float]] = []
            b_eq: list[float] = []
            for j, c in enumerate(constraints):
                if not isinstance(c, dict):
                    raise ValueError(f"constraints[{j}] must be an object")
                row = _coerce_float_list(
                    list(c.get("coefficients", [])),
                    f"constraints[{j}].coefficients",
                )
                if len(row) != n_vars:
                    return ToolResult(
                        succeeded=False,
                        error=(
                            f"constraints[{j}].coefficients has length {len(row)}; "
                            f"expected {n_vars}."
                        ),
                    )
                csense = str(c.get("sense", "<=")).strip()
                rhs = float(c.get("rhs", 0.0))
                if csense not in _SUPPORTED_CONSTRAINT_SENSES:
                    return ToolResult(
                        succeeded=False,
                        error=(
                            f"constraints[{j}].sense must be one of "
                            f"{_SUPPORTED_CONSTRAINT_SENSES}; got {csense!r}."
                        ),
                    )
                if csense == "<=":
                    A_ub.append(row)
                    b_ub.append(rhs)
                elif csense == ">=":
                    A_ub.append([-x for x in row])
                    b_ub.append(-rhs)
                else:  # "=="
                    A_eq.append(row)
                    b_eq.append(rhs)
        except ValueError as e:
            return ToolResult(succeeded=False, error=str(e))

        # linprog minimizes c·x; for maximize, negate c, then negate the result.
        c_vec = obj_coeffs if sense == "minimize" else [-x for x in obj_coeffs]

        try:
            from scipy.optimize import linprog
        except ImportError as e:
            return ToolResult(
                succeeded=False, error=f"scipy not available: {e}"
            )

        kwargs: dict[str, Any] = {"c": c_vec, "bounds": bounds, "method": "highs"}
        if A_ub:
            kwargs["A_ub"] = A_ub
            kwargs["b_ub"] = b_ub
        if A_eq:
            kwargs["A_eq"] = A_eq
            kwargs["b_eq"] = b_eq

        result = linprog(**kwargs)

        # Status semantics from HiGHS via scipy:
        #   0 optimal, 2 infeasible, 3 unbounded, 4 numerical issues, others
        status = int(result.status)
        message = str(result.message)

        if status == 0 and result.success:
            x = [float(v) for v in result.x]
            obj_val = float(result.fun) if sense == "minimize" else -float(result.fun)
            solution = dict(zip(var_names, x))
            normalized = {
                "kind": "linear",
                "sense": sense,
                "status": status,
                "status_message": message,
                "solution": solution,
                "objective_value": obj_val,
                "n_variables": n_vars,
                "n_constraints": len(constraints),
                "solver": "scipy.linprog (HiGHS)",
            }
            return ToolResult(
                evidence=[
                    build_evidence(
                        claim_id=request.claim.id,
                        source_type=SourceType.OPTIMIZER,
                        source_name="scipy.linprog",
                        quote_or_result=(
                            f"LP {sense}: objective = {obj_val:.6g}; "
                            + ", ".join(
                                f"{name}={val:.6g}" for name, val in solution.items()
                            )
                            + f". Status: {message}"
                        ),
                        normalized_result=normalized,
                        supports_claim=SupportStatus.SUPPORTS,
                        source_quality=SourceQuality.AUTHORITATIVE,
                        freshness=FreshnessStatus.NOT_TIME_SENSITIVE,
                        notes=(
                            "Optimum is conditional on the supplied objective and "
                            "constraints. 'Best' relative to a different model may differ."
                        ),
                    )
                ]
            )

        # status 2 = infeasible, 3 = unbounded, etc.
        infeasible_or_unbounded = status in (2, 3)
        return ToolResult(
            succeeded=False,
            error=(
                f"LP did not reach an optimal solution. Status {status}: {message}."
            ),
            evidence=[
                build_evidence(
                    claim_id=request.claim.id,
                    source_type=SourceType.OPTIMIZER,
                    source_name="scipy.linprog",
                    quote_or_result=f"LP terminated without optimum. Status {status}: {message}",
                    normalized_result={
                        "kind": "linear",
                        "sense": sense,
                        "status": status,
                        "status_message": message,
                        "n_variables": n_vars,
                        "n_constraints": len(constraints),
                    },
                    # Infeasible / unbounded actively refute any "X is the
                    # optimal allocation" claim — there is no optimum in the
                    # supplied model.
                    supports_claim=(
                        SupportStatus.CONTRADICTS
                        if infeasible_or_unbounded
                        else SupportStatus.INSUFFICIENT
                    ),
                    source_quality=SourceQuality.AUTHORITATIVE,
                    freshness=FreshnessStatus.NOT_TIME_SENSITIVE,
                    notes=(
                        "Infeasible model: constraints have no common feasible point; "
                        "no allocation satisfies them."
                        if status == 2
                        else "Unbounded model: objective improves without limit; check constraints."
                        if status == 3
                        else "Solver did not converge."
                    ),
                )
            ],
        )


_: Tool = LinearOptimizerTool()
