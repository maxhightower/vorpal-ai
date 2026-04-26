"""Search for contradictions or stale-vs-fresh conflicts within the evidence set.

The MVP applies two cheap, deterministic checks:

1. Recency dominance: when two pieces of evidence reference the same source
   family (e.g. policy documents) but disagree, the one with a more recent
   ``effective_date`` is flagged as authoritative and the other is marked as
   ``CONTRADICTS``.
2. Lexical opposition: a small list of antonym pairs (e.g. "must"/"may",
   "permitted"/"prohibited", "remote"/"in office") is used to flag opposed
   sentences citing the same claim.

Both are heuristics; they should be replaced with structured policy semantics
or NLI models when those are wired in.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date

from ..domain.evidence import Evidence, SupportStatus

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


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _has_opposition(a: str, b: str) -> bool:
    a = a.lower()
    b = b.lower()
    for left, right in _OPPOSITION_PAIRS:
        if (left in a and right in b) or (right in a and left in b):
            return True
    return False


def check_contradictions(evidence: list[Evidence]) -> tuple[list[Evidence], list[str]]:
    """Return updated evidence with contradiction flags, plus a list of human-readable notes."""

    # Index evidence by claim. Contradictions only matter when they affect the
    # same claim — different claims are allowed to disagree.
    by_claim: dict[str, list[Evidence]] = defaultdict(list)
    for e in evidence:
        by_claim[e.claim_id].append(e)

    notes: list[str] = []
    updated: list[Evidence] = []
    handled: set[str] = set()

    for claim_id, group in by_claim.items():
        if len(group) < 2:
            updated.extend(group)
            continue

        flagged: dict[str, Evidence] = {}
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if not _has_opposition(a.quote_or_result, b.quote_or_result):
                    continue

                # Choose which side to mark as CONTRADICTS based on freshness.
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
                    f"{loser.source_name!r} disagrees with {winner.source_name!r}."
                )
                handled.add(loser.id)
                handled.add(winner.id)

        for e in group:
            if e.id in flagged:
                updated.append(flagged[e.id])
            else:
                updated.append(e)

    return updated, notes
