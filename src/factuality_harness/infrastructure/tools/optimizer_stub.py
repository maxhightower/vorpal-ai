"""Optimizer stub. Real integration (linprog, OR-Tools) lives behind this interface."""

from __future__ import annotations

from ...domain.evidence import SourceType, SupportStatus
from .base import Tool, ToolRequest, ToolResult, build_evidence


class OptimizerStub:
    name = "optimizer"

    def run(self, request: ToolRequest) -> ToolResult:
        return ToolResult(
            evidence=[
                build_evidence(
                    claim_id=request.claim.id,
                    source_type=SourceType.OPTIMIZER,
                    source_name="optimizer_stub",
                    quote_or_result=(
                        "Optimizer not configured. Optimization claims require an "
                        "explicit objective function and constraints."
                    ),
                    supports_claim=SupportStatus.INSUFFICIENT,
                )
            ],
            metadata={"stub": True},
        )


_: Tool = OptimizerStub()
