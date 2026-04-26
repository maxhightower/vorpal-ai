"""Map epistemic types to verification tools.

The router does not execute tools — it produces a list of ``VerificationTask``
records that the evidence builder consumes. Keeping routing and execution apart
makes both auditable and lets us run tasks in parallel later.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..domain.claims import Claim
from ..domain.epistemic_types import EpistemicType


class VerificationTask(BaseModel):
    claim: Claim
    tool_names: list[str] = Field(default_factory=list)
    rationale: str = ""


# Default routing table. Domain modules may augment this per-claim.
_DEFAULT_ROUTES: dict[EpistemicType, list[str]] = {
    # SQL and Python tools no-op cleanly when their context isn't supplied,
    # so listing them here makes them auto-activate whenever the request
    # provides ``sql=`` or ``code=`` in extra_context.
    EpistemicType.NUMERICAL: ["calculator", "sql_executor", "python_executor"],
    EpistemicType.DIRECT_FACT: ["local_document_retriever", "sql_executor"],
    EpistemicType.LOGICAL: ["rule_engine", "theorem_prover", "python_executor"],
    EpistemicType.PROCEDURAL: ["rule_engine", "local_document_retriever"],
    EpistemicType.CAUSAL: ["causal_model"],
    EpistemicType.PREDICTIVE: ["causal_model"],
    EpistemicType.OPTIMIZATION: ["optimizer", "python_executor"],
    EpistemicType.INTERPRETIVE: ["local_document_retriever"],
    EpistemicType.SPECULATIVE: [],  # nothing verifies pure speculation
    EpistemicType.UNKNOWN: ["local_document_retriever"],
}


class ToolRouter:
    def __init__(self, routes: dict[EpistemicType, list[str]] | None = None) -> None:
        self._routes = routes or _DEFAULT_ROUTES

    def route(self, claim: Claim) -> VerificationTask:
        tools = list(self._routes.get(claim.epistemic_type, []))
        rationale = (
            f"Routed {claim.epistemic_type.value} claim to: "
            + (", ".join(tools) if tools else "<none — speculation>")
        )
        return VerificationTask(claim=claim, tool_names=tools, rationale=rationale)

    def route_all(self, claims: list[Claim]) -> list[VerificationTask]:
        return [self.route(c) for c in claims]
