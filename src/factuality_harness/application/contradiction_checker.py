"""Detect contradictions inside the evidence set for a single claim.

Architecture:

  ContradictionDetector (protocol)
    └── decides whether two strings contradict each other

  check_contradictions (the orchestrator)
    └── walks all same-claim evidence pairs, asks the detector,
        then applies recency-dominance to pick which side is marked
        as CONTRADICTS.

The orchestrator's job — pairing, recency tie-breaking, evidence
mutation, audit notes — is the same regardless of which detector is
used. Swapping detectors changes only *how* the binary
"do-these-disagree?" decision is made.

Detectors:

 - ``LexicalContradictionDetector`` (default): a small antonym-pair
   table. Cheap, deterministic, catches the majority of policy
   conflicts that disagree on common words.
 - ``LLMContradictionDetector`` (opt-in): asks an LLM to do a NLI
   (entailment / contradiction / neutral) classification on each pair.
   Output is parsed to a single label; any failure falls back to the
   lexical detector. This catches non-lexical contradictions —
   "must work in office three days" vs "remote-first by default" —
   that the antonym table misses.

Both implement the ``ContradictionDetector`` protocol so the orchestrator
and tests don't care which one is in use.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Protocol

from ..domain.evidence import Evidence, SupportStatus
from ..infrastructure.llm.base import LLM, LLMMessage, LLMRequest


# ---------------------------------------------------------------------------
# Detector protocol + implementations
# ---------------------------------------------------------------------------


class ContradictionDetector(Protocol):
    def is_contradicting(self, a: str, b: str) -> bool: ...


_OPPOSITION_PAIRS: list[tuple[str, str]] = [
    ("must ", "may "),
    ("must ", "need not "),
    ("required", "optional"),
    ("permitted", "prohibited"),
    ("allowed", "not allowed"),
    ("allowed", "forbidden"),
    ("in office", "remote"),
    ("in-office", "remote"),
    ("five days per week", "three days"),
    ("full-time", "part-time"),
    ("must work in office", "may work remotely"),
]


class LexicalContradictionDetector:
    """Antonym-pair table. Cheap and deterministic; misses non-lexical
    contradictions. Default detector and the fallback for the LLM path."""

    def is_contradicting(self, a: str, b: str) -> bool:
        a_lower = a.lower()
        b_lower = b.lower()
        for left, right in _OPPOSITION_PAIRS:
            if (left in a_lower and right in b_lower) or (
                right in a_lower and left in b_lower
            ):
                return True
        return False


_NLI_SYSTEM_PROMPT = """\
You are a strict natural-language-inference classifier.

Given two short text passages A and B, decide whether they CONTRADICT each
other. Two passages contradict when they cannot both be true for the same
subject at the same time.

Respond with EXACTLY ONE of these tokens, on a single line, with no prose:
  CONTRADICT
  ENTAIL
  NEUTRAL

Definitions:
  CONTRADICT - A and B make incompatible claims about the same subject.
  ENTAIL     - A implies B, or B implies A, with no conflict.
  NEUTRAL    - A and B address different subjects, or are compatible.

Be conservative. Only label CONTRADICT when the conflict is clear and the
two passages address the same subject. Do NOT label CONTRADICT for
differences in tone, scope, or specificity alone.
"""


class LLMContradictionDetector:
    """LLM-backed NLI classifier with a hard lexical fallback.

    The LLM sees only the two passages — never the claim text or any other
    context — to keep the call cheap and prompt-injection-resistant.
    """

    def __init__(
        self,
        *,
        llm: LLM,
        fallback: ContradictionDetector | None = None,
    ) -> None:
        self._llm = llm
        self._fallback = fallback or LexicalContradictionDetector()

    def is_contradicting(self, a: str, b: str) -> bool:
        if not a.strip() or not b.strip():
            return False
        prompt = f"A:\n{a.strip()}\n\nB:\n{b.strip()}\n\nLabel:"
        try:
            response = self._llm.complete(
                LLMRequest(
                    messages=[
                        LLMMessage(role="system", content=_NLI_SYSTEM_PROMPT),
                        LLMMessage(role="user", content=prompt),
                    ],
                    temperature=0.0,
                    max_tokens=10,
                )
            )
        except Exception:
            return self._fallback.is_contradicting(a, b)

        label = (response.text or "").strip().upper()
        # Strip code fences / bullets / quotes / trailing punctuation.
        label = label.lstrip("`*-•·\"' ").rstrip("`*\"' .,;:")
        if "CONTRADICT" in label:
            return True
        if "ENTAIL" in label or "NEUTRAL" in label:
            return False
        # Couldn't parse a label -> fall back so we don't silently miss
        # contradictions when the LLM misbehaves.
        return self._fallback.is_contradicting(a, b)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def check_contradictions(
    evidence: list[Evidence],
    *,
    detector: ContradictionDetector | None = None,
) -> tuple[list[Evidence], list[str]]:
    """Return updated evidence with contradiction flags, plus a list of
    human-readable notes.

    Same-claim evidence pairs are passed to ``detector.is_contradicting``;
    when it returns True, the side with the older ``effective_date`` (or
    the second one if dates are missing) is marked ``CONTRADICTS``.
    """
    detector = detector or LexicalContradictionDetector()

    # Index evidence by claim. Contradictions only matter when they affect
    # the same claim — different claims are allowed to disagree.
    by_claim: dict[str, list[Evidence]] = defaultdict(list)
    for e in evidence:
        by_claim[e.claim_id].append(e)

    notes: list[str] = []
    updated: list[Evidence] = []

    for claim_id, group in by_claim.items():
        if len(group) < 2:
            updated.extend(group)
            continue

        flagged: dict[str, Evidence] = {}
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if not detector.is_contradicting(a.quote_or_result, b.quote_or_result):
                    continue

                a_date = _parse_date(
                    (a.normalized_result or {}).get("effective_date")
                )
                b_date = _parse_date(
                    (b.normalized_result or {}).get("effective_date")
                )

                if a_date and b_date:
                    if a_date < b_date:
                        loser, winner = a, b
                    elif b_date < a_date:
                        loser, winner = b, a
                    else:
                        loser, winner = a, b  # tie: arbitrary
                else:
                    loser, winner = a, b

                losing = loser.model_copy(
                    update={
                        "supports_claim": SupportStatus.CONTRADICTS,
                        "notes": (
                            (loser.notes + " | " if loser.notes else "")
                            + f"Contradicted by {winner.source_name}."
                        ),
                    }
                )
                flagged[loser.id] = losing
                notes.append(
                    f"Contradiction on claim {claim_id}: "
                    f"{loser.source_name!r} disagrees with "
                    f"{winner.source_name!r}."
                )

        for e in group:
            updated.append(flagged.get(e.id, e))

    return updated, notes
