"""Minimal rule-engine tool.

Procedural claims need concrete policy text to be verifiable. The rule engine
takes a list of rule documents (loaded from the request context or supplied at
construction time), finds the most relevant rule by keyword overlap, and emits
that rule as evidence. It does NOT decide whether the rule supports or
contradicts the claim — the contradiction checker does that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ...domain.evidence import FreshnessStatus, SourceQuality, SourceType, SupportStatus
from .base import Tool, ToolRequest, ToolResult, build_evidence


@dataclass
class RuleDocument:
    name: str
    text: str
    effective_date: str | None = None


_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z\-']+")


def _tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(s) if len(t) > 2}


class RuleEngineTool:
    name = "rule_engine"

    def __init__(self, rules: list[RuleDocument] | None = None) -> None:
        self._rules: list[RuleDocument] = list(rules or [])

    def add_rule(self, rule: RuleDocument) -> None:
        self._rules.append(rule)

    def run(self, request: ToolRequest) -> ToolResult:
        claim_tokens = _tokens(request.claim.text)
        rules: list[RuleDocument] = list(self._rules)

        # Allow per-call rule injection through the request context.
        ctx_rules = request.context.get("rules") or []
        for r in ctx_rules:
            if isinstance(r, RuleDocument):
                rules.append(r)
            elif isinstance(r, dict):
                rules.append(
                    RuleDocument(
                        name=r.get("name", "rule"),
                        text=r.get("text", ""),
                        effective_date=r.get("effective_date"),
                    )
                )

        if not rules:
            return ToolResult(
                succeeded=False,
                error="No policy/rule documents available for this procedural claim.",
            )

        scored: list[tuple[float, RuleDocument]] = []
        for rule in rules:
            overlap = len(claim_tokens & _tokens(rule.text))
            if overlap > 0:
                scored.append((overlap, rule))

        if not scored:
            return ToolResult(
                succeeded=False,
                error="No rule document matched any keywords in the claim.",
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        evidence = []
        for score, rule in scored[:3]:  # cap at top 3 to keep table compact
            evidence.append(
                build_evidence(
                    claim_id=request.claim.id,
                    source_type=SourceType.RULE_ENGINE,
                    source_name=rule.name,
                    source_uri=None,
                    quote_or_result=rule.text,
                    normalized_result={
                        "rule_name": rule.name,
                        "effective_date": rule.effective_date,
                        "match_score": score,
                    },
                    supports_claim=SupportStatus.PARTIALLY_SUPPORTS,
                    source_quality=SourceQuality.AUTHORITATIVE,
                    freshness=(
                        FreshnessStatus.UNDATED
                        if rule.effective_date is None
                        else FreshnessStatus.CURRENT
                    ),
                    notes=f"Keyword overlap score: {score}",
                )
            )
        return ToolResult(evidence=evidence)


_: Tool = RuleEngineTool()
