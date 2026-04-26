"""Generate a draft answer strictly from the evidence table.

The MVP uses a deterministic template generator. The point is to make sure no
unsupported claim slips into the draft. A real LLM-backed generator can be
plugged in via the same interface, but the post-generation verifier remains
the safety net.
"""

from __future__ import annotations

from typing import Protocol

from ..domain.confidence import ConfidenceLevel
from ..domain.epistemic_types import EpistemicType
from ..domain.evidence import EvidenceTable
from ..domain.verdicts import ClaimVerdict, Verdict


_VERDICT_PHRASES: dict[Verdict, str] = {
    Verdict.VERIFIED: "is verified",
    Verdict.SUPPORTED: "is supported by evidence",
    Verdict.PARTIALLY_SUPPORTED: "is partially supported",
    Verdict.COMPUTED: "was computed deterministically",
    Verdict.FORMALLY_PROVEN: "is formally proven",
    Verdict.UNSUPPORTED: "is not supported by available evidence",
    Verdict.CONTRADICTED: "is contradicted by available evidence",
    Verdict.UNCLEAR: "cannot be determined from available evidence",
    Verdict.SPECULATIVE: "is speculative",
}


class DraftGenerator(Protocol):
    def generate(self, table: EvidenceTable) -> str: ...


class TemplateDraftGenerator:
    """Compose a draft from claim verdicts using clear, factual templates."""

    def generate(self, table: EvidenceTable) -> str:
        verdicts: list[ClaimVerdict] = list(table.verdicts)
        if not verdicts:
            return "No verifiable claims were extracted from the question."

        sections: list[str] = []
        for v in verdicts:
            claim = table.claim_by_id(v.claim_id)
            if claim is None:
                continue
            section = self._format_claim(claim, v, table)
            sections.append(section)

        # Caveats block: surface anything that needs care.
        caveats = self._format_caveats(verdicts, table)

        body = "\n\n".join(sections)
        return body + ("\n\n" + caveats if caveats else "")

    def _format_claim(self, claim, verdict: ClaimVerdict, table: EvidenceTable) -> str:
        phrase = _VERDICT_PHRASES.get(verdict.verdict, "has an unknown verdict")
        evidence_lines: list[str] = []
        for eid in verdict.evidence_ids:
            ev = next((e for e in table.evidence if e.id == eid), None)
            if ev is None:
                continue
            evidence_lines.append(
                f"  - [{ev.source_type.value}/{ev.source_quality.value}] "
                f"{ev.source_name}: {ev.quote_or_result}"
            )

        type_qualifier = ""
        if claim.epistemic_type == EpistemicType.PREDICTIVE:
            type_qualifier = " (prediction — uncertainty applies)"
        elif claim.epistemic_type == EpistemicType.CAUSAL:
            type_qualifier = " (causal — correlation alone is insufficient)"
        elif claim.epistemic_type == EpistemicType.SPECULATIVE:
            type_qualifier = " (speculative — not a factual assertion)"
        elif claim.epistemic_type == EpistemicType.OPTIMIZATION:
            type_qualifier = " (requires explicit objective and constraints)"

        head = f"- Claim: {claim.text}{type_qualifier}"
        body_lines = [
            f"  Verdict: {verdict.verdict.value} (confidence: {verdict.confidence.value}). "
            f"This claim {phrase}.",
            f"  Rationale: {verdict.rationale}",
        ]
        if verdict.assumptions:
            body_lines.append("  Assumptions: " + "; ".join(verdict.assumptions))
        if verdict.limitations:
            body_lines.append("  Limitations: " + "; ".join(verdict.limitations))
        if evidence_lines:
            body_lines.append("  Evidence:")
            body_lines.extend(evidence_lines)
        return "\n".join([head, *body_lines])

    def _format_caveats(self, verdicts: list[ClaimVerdict], table: EvidenceTable) -> str:
        caveats: list[str] = []
        for v in verdicts:
            claim = table.claim_by_id(v.claim_id)
            if claim is None:
                continue
            if v.verdict in (Verdict.UNSUPPORTED, Verdict.UNCLEAR, Verdict.CONTRADICTED):
                caveats.append(
                    f"  - {claim.text}: {v.verdict.value.lower()} ({v.rationale})"
                )
            elif v.confidence == ConfidenceLevel.LOW:
                caveats.append(f"  - {claim.text}: low confidence — {v.rationale}")
        if not caveats:
            return ""
        return "Caveats and uncertainties:\n" + "\n".join(caveats)
