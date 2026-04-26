"""Causal model stub.

The harness's policy is that causal claims default to UNSUPPORTED unless
real experimental, quasi-experimental, or causal-model evidence is provided.
This stub explicitly emits that no causal evidence is available, which keeps
the verifier honest.
"""

from __future__ import annotations

from ...domain.evidence import SourceQuality, SourceType, SupportStatus
from .base import Tool, ToolRequest, ToolResult, build_evidence


class CausalModelStub:
    name = "causal_model"

    def run(self, request: ToolRequest) -> ToolResult:
        return ToolResult(
            evidence=[
                build_evidence(
                    claim_id=request.claim.id,
                    source_type=SourceType.CAUSAL_MODEL,
                    source_name="causal_model_stub",
                    quote_or_result=(
                        "No experimental, quasi-experimental, or causal-model "
                        "evidence available for this claim."
                    ),
                    supports_claim=SupportStatus.INSUFFICIENT,
                    source_quality=SourceQuality.UNKNOWN,
                    notes=(
                        "Correlation alone cannot establish causation. Provide an "
                        "RCT, diff-in-diff, instrumental variable, or causal graph."
                    ),
                )
            ],
            metadata={"stub": True},
        )


_: Tool = CausalModelStub()
