"""Classify each claim by epistemic type.

Phase 2 uses a deterministic keyword/regex classifier. The interface allows an
LLM-backed classifier to be plugged in later. The rules deliberately err on the
side of stricter (e.g. SPECULATIVE, CAUSAL) types because over-permissive
classification leads to overclaiming downstream.
"""

from __future__ import annotations

import re
from typing import Protocol

from ..domain.claims import Claim, ClaimStatus
from ..domain.epistemic_types import EpistemicType

# Keep patterns simple and inspectable. Order matters: earlier patterns win.
_RULES: list[tuple[EpistemicType, re.Pattern[str]]] = [
    (
        EpistemicType.NUMERICAL,
        re.compile(
            r"\b(percent|percentage|%|increase from|decrease from|ratio|average|"
            r"sum of|total of|compute|calculate|how much|how many|growth rate|"
            r"\d+\s*[\+\-\*/x×]\s*\d+|"
            r"\d+(\.\d+)?\s*(to|->|→)\s*\d+(\.\d+)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        EpistemicType.CAUSAL,
        re.compile(
            r"\b(caus(?:e|ed|ing|es)|because of|due to|led to|drove|driver of|"
            r"impact of|effect of|attribut(?:e|ion)|why did|reason (?:for|behind))\b",
            re.IGNORECASE,
        ),
    ),
    (
        EpistemicType.PREDICTIVE,
        re.compile(
            r"\b(will|forecast|projected|expected to|next quarter|next year|"
            r"future|going to|by 20\d\d)\b",
            re.IGNORECASE,
        ),
    ),
    (
        EpistemicType.OPTIMIZATION,
        re.compile(
            r"\b(best|optimal|optimi[sz]e|maximi[sz]e|minimi[sz]e|allocate|"
            r"trade-?off|cheapest|fastest|most efficient)\b",
            re.IGNORECASE,
        ),
    ),
    (
        EpistemicType.PROCEDURAL,
        re.compile(
            r"\b(allowed|permitted|compliant|complian(?:ce|t)|violat(?:e|ion)|"
            r"under (?:the )?policy|per (?:the )?policy|regulation|rule|"
            r"is it legal|are we allowed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        EpistemicType.LOGICAL,
        re.compile(
            r"\b(does .{0,30}follow|prove|implies|imply|entails?|if .* then|"
            r"contradiction|valid argument|logically|premises?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        EpistemicType.SPECULATIVE,
        re.compile(
            r"\b(could|might|maybe|hypothetically|speculate|imagine if|"
            r"what if)\b",
            re.IGNORECASE,
        ),
    ),
    (
        EpistemicType.INTERPRETIVE,
        re.compile(
            r"\b(summari[sz]e|interpret|assessment|qualitative|judgment|"
            r"strategy|positioning|narrative|explain why)\b",
            re.IGNORECASE,
        ),
    ),
    (
        EpistemicType.DIRECT_FACT,
        re.compile(
            r"\b(who|what (?:is|was|were|are)|where|when|which|"
            r"how many of|name of|current|today|as of|capital of|"
            r"founded|headquartered|how was|tell me about)\b",
            re.IGNORECASE,
        ),
    ),
]


# Flags to set on the claim alongside the type.
_FLAG_RULES: dict[EpistemicType, dict[str, bool]] = {
    EpistemicType.NUMERICAL: {"requires_computation": True},
    EpistemicType.CAUSAL: {"requires_causal_inference": True},
    EpistemicType.LOGICAL: {"requires_formal_verification": True},
    EpistemicType.PREDICTIVE: {"requires_current_info": True},
    EpistemicType.DIRECT_FACT: {"requires_current_info": True},
}


class ClaimClassifier(Protocol):
    def classify(self, claim: Claim) -> Claim: ...

    def classify_all(self, claims: list[Claim]) -> list[Claim]: ...


class RuleBasedClaimClassifier:
    def classify(self, claim: Claim) -> Claim:
        text = claim.text
        for etype, pattern in _RULES:
            if pattern.search(text):
                flags = _FLAG_RULES.get(etype, {})
                return claim.model_copy(
                    update={
                        "epistemic_type": etype,
                        "status": ClaimStatus.CLASSIFIED,
                        **flags,
                    }
                )
        # Fallback: a question of unknown shape becomes DIRECT_FACT only if it
        # *looks* like a direct factual question; otherwise UNKNOWN.
        return claim.model_copy(
            update={
                "epistemic_type": EpistemicType.UNKNOWN,
                "status": ClaimStatus.CLASSIFIED,
            }
        )

    def classify_all(self, claims: list[Claim]) -> list[Claim]:
        return [self.classify(c) for c in claims]
