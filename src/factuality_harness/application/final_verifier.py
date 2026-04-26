"""Post-draft verification.

After the draft is composed, extract assertions from it and check each one
against the evidence table. Any assertion that is not backed by a verdict of
``SUPPORTED``/``VERIFIED``/``COMPUTED``/``FORMALLY_PROVEN`` is removed or
qualified before the final answer is returned.

The MVP implements:

 - A claim extractor that splits the draft into bullet-line claims (the
   template generator already labels them with "Claim:" prefixes).
 - A verifier that maps each draft claim back to an upstream verdict by id.
 - A revision pass that drops or qualifies unsupported assertions and writes
   the final answer.
"""

from __future__ import annotations

import re
from typing import Protocol

from ..domain.claims import Claim, ClaimStatus
from ..domain.confidence import ConfidenceLevel
from ..domain.epistemic_types import EpistemicType
from ..domain.evidence import EvidenceTable
from ..domain.verdicts import ClaimVerdict, FinalAnswer, Verdict

_CLAIM_LINE = re.compile(r"^\s*-\s*Claim:\s*(?P<text>.+?)\s*$", re.MULTILINE)


class DraftClaimExtractor(Protocol):
    def extract(self, draft: str, table: EvidenceTable) -> list[Claim]: ...


class TemplateDraftClaimExtractor:
    """Extract claims emitted by ``TemplateDraftGenerator``.

    The generator writes lines beginning with ``- Claim:``; we re-attach the
    text to the original claim object so the final verifier can use the upstream
    verdict directly. If a draft line cannot be matched to an upstream claim,
    we treat it as a *new* unsupported assertion.
    """

    def extract(self, draft: str, table: EvidenceTable) -> list[Claim]:
        extracted: list[Claim] = []
        for match in _CLAIM_LINE.finditer(draft):
            raw_text = match.group("text").strip()
            # Strip type-qualifier suffixes (e.g. " (prediction — ...)") for
            # robust matching against the original claim text.
            stripped = re.sub(r"\s*\(.+?\)\s*$", "", raw_text).strip()

            original = next(
                (c for c in table.claims if c.text.strip() == stripped),
                None,
            )
            if original is not None:
                extracted.append(original.with_status(ClaimStatus.VERDICT_ASSIGNED))
            else:
                extracted.append(
                    Claim(
                        text=stripped,
                        parent_question=table.original_question,
                        epistemic_type=EpistemicType.UNKNOWN,
                        status=ClaimStatus.PROPOSED,
                    )
                )
        return extracted


def verify_draft_claims(
    draft_claims: list[Claim], table: EvidenceTable
) -> list[ClaimVerdict]:
    """Map each draft claim to its upstream verdict, or mark UNSUPPORTED."""

    verdicts_by_claim_id: dict[str, ClaimVerdict] = {
        v.claim_id: v for v in table.verdicts
    }
    out: list[ClaimVerdict] = []
    for c in draft_claims:
        v = verdicts_by_claim_id.get(c.id)
        if v is not None:
            out.append(v)
        else:
            out.append(
                ClaimVerdict(
                    claim_id=c.id,
                    verdict=Verdict.UNSUPPORTED,
                    confidence=ConfidenceLevel.LOW,
                    rationale=(
                        "Draft introduced a claim that does not correspond to any "
                        "verified upstream claim."
                    ),
                )
            )
    return out


_VERIFIED_VERDICTS = {
    Verdict.VERIFIED,
    Verdict.SUPPORTED,
    Verdict.COMPUTED,
    Verdict.FORMALLY_PROVEN,
}


def revise_answer(
    *,
    question: str,
    draft: str,
    draft_verdicts: list[ClaimVerdict],
    table: EvidenceTable,
    audit_id: str,
) -> FinalAnswer:
    """Strip or qualify unsupported claims and produce the final answer."""

    unsupported_or_uncertain: list[str] = []
    revised_lines = draft.splitlines()
    revised: list[str] = []

    # Index draft verdicts by claim text for line-level lookup.
    verdicts_by_text: dict[str, ClaimVerdict] = {}
    for v in draft_verdicts:
        c = table.claim_by_id(v.claim_id)
        if c is not None:
            verdicts_by_text[c.text.strip()] = v

    skip_block = False
    block_qualifier: str | None = None
    for line in revised_lines:
        m = _CLAIM_LINE.match(line)
        if m:
            stripped = re.sub(r"\s*\(.+?\)\s*$", "", m.group("text").strip()).strip()
            v = verdicts_by_text.get(stripped)
            if v is None:
                # No matching verdict — treat as unsupported and drop the block.
                unsupported_or_uncertain.append(stripped)
                skip_block = True
                block_qualifier = None
                continue
            if v.verdict in _VERIFIED_VERDICTS and v.confidence != ConfidenceLevel.LOW:
                skip_block = False
                block_qualifier = None
                revised.append(line)
                continue
            # Keep the claim block but prefix a qualifier; record uncertainty.
            unsupported_or_uncertain.append(stripped)
            skip_block = False
            block_qualifier = (
                f"  [QUALIFIED: {v.verdict.value} — confidence {v.confidence.value}]"
            )
            revised.append(line)
            revised.append(block_qualifier)
            continue

        # Continuation lines belong to the previous claim block. Skip them when
        # the block was dropped; otherwise keep them.
        if skip_block:
            continue
        revised.append(line)

    # Build the confidence summary.
    counts: dict[str, int] = {}
    for v in draft_verdicts:
        counts[v.verdict.value] = counts.get(v.verdict.value, 0) + 1
    summary = ", ".join(f"{k}: {n}" for k, n in sorted(counts.items()))
    if not summary:
        summary = "No verdicts."

    final_text = "\n".join(line for line in revised if line is not None).strip()
    if unsupported_or_uncertain:
        # Mention the count only. The claims themselves are returned in the
        # structured ``unsupported_or_uncertain_claims`` field so callers can
        # render them however they want, without re-asserting their text in
        # the answer body.
        final_text += (
            f"\n\nNote: {len(unsupported_or_uncertain)} claim(s) were "
            "unsupported or uncertain and have been qualified or removed. "
            "See `unsupported_or_uncertain_claims` for details."
        )

    return FinalAnswer(
        answer=final_text,
        confidence_summary=summary,
        unsupported_or_uncertain_claims=unsupported_or_uncertain,
        evidence_table=table,
        audit_id=audit_id,
    )
