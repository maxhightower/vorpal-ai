"""Decompose a natural-language question into atomic claims.

The MVP uses a rule-based decomposer that splits on conjunctions, comma-clauses,
and question markers. The interface is designed so an LLM-backed decomposer can
be substituted later without changing callers.
"""

from __future__ import annotations

import re
from typing import Protocol

from ..domain.claims import Claim, ClaimStatus

# Conjunctions and connectives that often join independent claims.
_SPLIT_PATTERN = re.compile(
    r"\s+(?:and also|and then|and|;|,\s*and|but also|but|however|while|whereas)\s+",
    flags=re.IGNORECASE,
)

# Question pivots within a single sentence (e.g. "X, and did Y?").
_QUESTION_PIVOT = re.compile(r"(?<=[\.\?])\s+", flags=re.IGNORECASE)


class ClaimDecomposer(Protocol):
    def decompose(self, question: str) -> list[Claim]: ...


class RuleBasedClaimDecomposer:
    """Cheap, deterministic decomposer.

    It is intentionally conservative: when in doubt, it produces fewer, larger
    claims rather than over-splitting. Over-splitting tends to produce
    unverifiable fragments.
    """

    def decompose(self, question: str) -> list[Claim]:
        question = question.strip()
        if not question:
            return []

        # First split by sentence pivots, then by intra-sentence connectives.
        sentences = [s.strip() for s in _QUESTION_PIVOT.split(question) if s.strip()]
        fragments: list[str] = []
        for sentence in sentences:
            parts = [p.strip(" ?.") for p in _SPLIT_PATTERN.split(sentence) if p.strip(" ?.")]
            fragments.extend(parts if parts else [sentence.strip(" ?.")])

        # De-duplicate while preserving order.
        seen: set[str] = set()
        ordered: list[str] = []
        for f in fragments:
            key = f.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(f)

        # If the question only generates one fragment, the question itself is the claim.
        if not ordered:
            ordered = [question.strip(" ?.")]

        return [
            Claim(
                text=fragment,
                normalized_text=fragment.lower(),
                parent_question=question,
                status=ClaimStatus.PROPOSED,
            )
            for fragment in ordered
        ]
