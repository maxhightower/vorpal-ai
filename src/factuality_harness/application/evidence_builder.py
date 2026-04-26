"""Execute verification tasks and aggregate the evidence they emit.

The evidence builder is intentionally thin: it owns no domain knowledge. It
holds a registry of named tools and a retriever, dispatches each verification
task, and collects results. Anything smarter (verdict assignment, contradiction
search) lives in dedicated modules so each step can be inspected on its own.
"""

from __future__ import annotations

from typing import Any

from ..domain.audit import ToolCallRecord
from ..domain.claims import Claim, ClaimStatus
from ..domain.evidence import (
    Evidence,
    FreshnessStatus,
    SourceQuality,
    SourceType,
    SupportStatus,
)
from ..infrastructure.retrieval.base import Retriever
from ..infrastructure.tools.base import Tool, ToolRequest
from .router import VerificationTask


class EvidenceBuilder:
    def __init__(
        self,
        *,
        tools: dict[str, Tool] | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = dict(tools or {})
        self._retriever = retriever

    def register_tool(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def execute(
        self,
        tasks: list[VerificationTask],
        *,
        context: dict[str, Any] | None = None,
    ) -> tuple[list[Claim], list[Evidence], list[ToolCallRecord]]:
        context = context or {}
        all_evidence: list[Evidence] = []
        call_records: list[ToolCallRecord] = []
        updated_claims: list[Claim] = []

        for task in tasks:
            claim = task.claim
            produced: list[Evidence] = []

            for tool_name in task.tool_names:
                # Special case: the retriever is not registered as a tool but as
                # its own port — adapt it on the fly so routing stays uniform.
                if tool_name == "local_document_retriever":
                    if self._retriever is None:
                        call_records.append(
                            ToolCallRecord(
                                tool_name=tool_name,
                                inputs={"claim": claim.text},
                                succeeded=False,
                                error="No retriever configured.",
                            )
                        )
                        continue
                    results = self._retriever.retrieve(claim.text, top_k=3)
                    if not results:
                        call_records.append(
                            ToolCallRecord(
                                tool_name=tool_name,
                                inputs={"claim": claim.text},
                                outputs={"results": 0},
                            )
                        )
                        continue
                    for r in results:
                        produced.append(
                            Evidence(
                                claim_id=claim.id,
                                source_type=SourceType.LOCAL_DOCUMENT,
                                source_name=r.document.name,
                                source_uri=r.document.uri,
                                quote_or_result=r.snippet,
                                normalized_result={
                                    "score": r.score,
                                    "effective_date": r.document.effective_date,
                                },
                                supports_claim=SupportStatus.PARTIALLY_SUPPORTS,
                                source_quality=SourceQuality.SECONDARY,
                                freshness=(
                                    FreshnessStatus.CURRENT
                                    if r.document.effective_date
                                    else FreshnessStatus.UNDATED
                                ),
                            )
                        )
                    call_records.append(
                        ToolCallRecord(
                            tool_name=tool_name,
                            inputs={"claim": claim.text},
                            outputs={"results": len(results)},
                        )
                    )
                    continue

                tool = self._tools.get(tool_name)
                if tool is None:
                    call_records.append(
                        ToolCallRecord(
                            tool_name=tool_name,
                            inputs={"claim": claim.text},
                            succeeded=False,
                            error=f"Tool {tool_name!r} not registered.",
                        )
                    )
                    continue

                result = tool.run(ToolRequest(claim=claim, context=context))
                produced.extend(result.evidence)
                call_records.append(
                    ToolCallRecord(
                        tool_name=tool_name,
                        inputs={"claim": claim.text},
                        outputs={
                            "evidence_count": len(result.evidence),
                            **result.metadata,
                        },
                        succeeded=result.succeeded,
                        error=result.error,
                    )
                )

            all_evidence.extend(produced)
            updated_claims.append(
                claim.with_status(
                    ClaimStatus.EVIDENCE_GATHERED if produced else ClaimStatus.ROUTED
                )
            )

        return updated_claims, all_evidence, call_records
