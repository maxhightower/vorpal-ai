"""Theorem prover stub. Real integration (Z3, Lean) lives behind this interface."""

from __future__ import annotations

from ...domain.evidence import SourceType
from .base import Tool, ToolRequest, ToolResult, build_evidence


class TheoremProverStub:
    name = "theorem_prover"

    def run(self, request: ToolRequest) -> ToolResult:
        return ToolResult(
            evidence=[
                build_evidence(
                    claim_id=request.claim.id,
                    source_type=SourceType.NONE,
                    source_name="theorem_prover_stub",
                    quote_or_result=(
                        "Theorem prover not configured. Logical claims should be "
                        "marked UNCLEAR until a prover is wired up."
                    ),
                )
            ],
            succeeded=True,
            metadata={"stub": True},
        )


_: Tool = TheoremProverStub()
