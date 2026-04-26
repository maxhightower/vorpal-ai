"""Assign verdicts and confidence to claims based on evidence + epistemic type.

The verdict rules in the spec are encoded explicitly here so they can be
read without hunting through unrelated logic:

 - Numerical claims need authoritative computation/structured-data evidence to
   reach VERIFIED/COMPUTED.
 - Causal claims default to UNSUPPORTED unless explicitly causal evidence
   exists.
 - Predictive claims must never be VERIFIED.
 - Speculative claims must never be VERIFIED or SUPPORTED without
   qualification — we always emit SPECULATIVE for them.
 - Procedural claims require applicable rule text.
 - Direct factual claims need source evidence.
 - Logical claims need formal-prover evidence to be FORMALLY_PROVEN.
 - Missing evidence yields UNSUPPORTED or UNCLEAR — never a guess.
"""

from __future__ import annotations

from ..domain.claims import Claim
from ..domain.confidence import ConfidenceLevel
from ..domain.epistemic_types import EpistemicType
from ..domain.evidence import Evidence, SourceQuality, SourceType, SupportStatus
from ..domain.verdicts import ClaimVerdict, Verdict


def _supports(e: Evidence) -> bool:
    return e.supports_claim == SupportStatus.SUPPORTS


def _contradicts(e: Evidence) -> bool:
    return e.supports_claim == SupportStatus.CONTRADICTS


def _partial(e: Evidence) -> bool:
    return e.supports_claim == SupportStatus.PARTIALLY_SUPPORTS


def _highest_quality(evs: list[Evidence]) -> SourceQuality:
    order = [
        SourceQuality.AUTHORITATIVE,
        SourceQuality.PRIMARY,
        SourceQuality.SECONDARY,
        SourceQuality.LOW_QUALITY,
        SourceQuality.UNKNOWN,
    ]
    for q in order:
        if any(e.source_quality == q for e in evs):
            return q
    return SourceQuality.UNKNOWN


def assign_verdict(claim: Claim, evidence: list[Evidence]) -> ClaimVerdict:
    evs = [e for e in evidence if e.claim_id == claim.id]

    # Speculative claims: never VERIFIED, never SUPPORTED without qualification.
    if claim.epistemic_type == EpistemicType.SPECULATIVE:
        return ClaimVerdict(
            claim_id=claim.id,
            verdict=Verdict.SPECULATIVE,
            confidence=ConfidenceLevel.LOW,
            rationale="Claim is speculative; cannot be verified.",
            evidence_ids=[e.id for e in evs],
            limitations=[
                "Speculative claims are reported as scenarios, not facts.",
            ],
        )

    # Predictive claims: never VERIFIED.
    if claim.epistemic_type == EpistemicType.PREDICTIVE:
        forecast_evs = [e for e in evs if e.source_type == SourceType.FORECAST_MODEL]
        if forecast_evs:
            return ClaimVerdict(
                claim_id=claim.id,
                verdict=Verdict.SUPPORTED,
                confidence=ConfidenceLevel.LOW,
                rationale="Predictive claim supported by a forecast model; treat with uncertainty.",
                evidence_ids=[e.id for e in forecast_evs],
                limitations=["Predictions are inherently uncertain; ranges should be reported."],
            )
        return ClaimVerdict(
            claim_id=claim.id,
            verdict=Verdict.UNCLEAR,
            confidence=ConfidenceLevel.LOW,
            rationale="No forecast model or scenario evidence is available for this prediction.",
            evidence_ids=[e.id for e in evs],
            limitations=["Predictions cannot be VERIFIED; require explicit assumptions."],
        )

    # Causal claims: only causal-model or experimental evidence counts.
    if claim.epistemic_type == EpistemicType.CAUSAL:
        causal_evs = [e for e in evs if e.source_type == SourceType.CAUSAL_MODEL]
        causal_supports = [e for e in causal_evs if _supports(e)]
        if causal_supports:
            return ClaimVerdict(
                claim_id=claim.id,
                verdict=Verdict.SUPPORTED,
                confidence=ConfidenceLevel.MEDIUM,
                rationale="Causal evidence available; effect supported by causal model.",
                evidence_ids=[e.id for e in causal_supports],
            )
        # A causal model ran but failed to reach a SUPPORTS verdict (e.g. an
        # A/B test with a non-significant result). Distinguish this from "no
        # causal model was run at all" so the rationale is accurate.
        if causal_evs:
            return ClaimVerdict(
                claim_id=claim.id,
                verdict=Verdict.UNSUPPORTED,
                confidence=ConfidenceLevel.LOW,
                rationale=(
                    "Causal model ran but did not return support for the effect "
                    "(e.g. result not statistically significant). Causation cannot "
                    "be claimed."
                ),
                evidence_ids=[e.id for e in evs],
                limitations=[
                    "Insufficient power, noisy effect, or genuine null result.",
                ],
            )
        return ClaimVerdict(
            claim_id=claim.id,
            verdict=Verdict.UNSUPPORTED,
            confidence=ConfidenceLevel.LOW,
            rationale=(
                "No experimental, quasi-experimental, or causal-model evidence "
                "available. Correlation does not establish causation."
            ),
            evidence_ids=[e.id for e in evs],
            limitations=[
                "Causal claims require RCT/diff-in-diff/IV/causal-graph evidence.",
            ],
        )

    # Numerical claims: need a calculator/computation result.
    if claim.epistemic_type == EpistemicType.NUMERICAL:
        comp = [
            e
            for e in evs
            if e.source_type == SourceType.COMPUTATION and _supports(e)
        ]
        if comp:
            return ClaimVerdict(
                claim_id=claim.id,
                verdict=Verdict.COMPUTED,
                confidence=ConfidenceLevel.HIGH,
                rationale="Result computed deterministically.",
                evidence_ids=[e.id for e in comp],
            )
        struct = [
            e
            for e in evs
            if e.source_type == SourceType.STRUCTURED_DATA
            and e.source_quality == SourceQuality.AUTHORITATIVE
            and _supports(e)
        ]
        if struct:
            return ClaimVerdict(
                claim_id=claim.id,
                verdict=Verdict.VERIFIED,
                confidence=ConfidenceLevel.HIGH,
                rationale="Value read from authoritative structured data.",
                evidence_ids=[e.id for e in struct],
            )
        return ClaimVerdict(
            claim_id=claim.id,
            verdict=Verdict.UNSUPPORTED,
            confidence=ConfidenceLevel.LOW,
            rationale="Numerical claim has no computation or structured-data support.",
            evidence_ids=[e.id for e in evs],
        )

    # Logical claims: need a theorem prover / rule engine for formal proof.
    if claim.epistemic_type == EpistemicType.LOGICAL:
        proven = [
            e for e in evs if e.source_type == SourceType.THEOREM_PROVER and _supports(e)
        ]
        if proven:
            return ClaimVerdict(
                claim_id=claim.id,
                verdict=Verdict.FORMALLY_PROVEN,
                confidence=ConfidenceLevel.HIGH,
                rationale="Proven by formal verifier.",
                evidence_ids=[e.id for e in proven],
            )
        return ClaimVerdict(
            claim_id=claim.id,
            verdict=Verdict.UNCLEAR,
            confidence=ConfidenceLevel.LOW,
            rationale="No formal proof available.",
            evidence_ids=[e.id for e in evs],
        )

    # Optimization claims: need an optimizer with explicit objective + constraints.
    if claim.epistemic_type == EpistemicType.OPTIMIZATION:
        opt = [e for e in evs if e.source_type == SourceType.OPTIMIZER and _supports(e)]
        if opt:
            return ClaimVerdict(
                claim_id=claim.id,
                verdict=Verdict.SUPPORTED,
                confidence=ConfidenceLevel.MEDIUM,
                rationale="Optimizer returned a feasible solution.",
                evidence_ids=[e.id for e in opt],
            )
        return ClaimVerdict(
            claim_id=claim.id,
            verdict=Verdict.UNCLEAR,
            confidence=ConfidenceLevel.LOW,
            rationale=(
                "Optimization claim requires an explicit objective function "
                "and constraints; none were provided."
            ),
            evidence_ids=[e.id for e in evs],
            limitations=["Cannot establish optimality without a formal model."],
        )

    # Procedural claims: need rule-engine or policy-document evidence.
    if claim.epistemic_type == EpistemicType.PROCEDURAL:
        rules = [
            e
            for e in evs
            if e.source_type in (SourceType.RULE_ENGINE, SourceType.LOCAL_DOCUMENT)
            and not _contradicts(e)
        ]
        contradictions = [e for e in evs if _contradicts(e)]
        if rules and not contradictions:
            return ClaimVerdict(
                claim_id=claim.id,
                verdict=Verdict.SUPPORTED,
                confidence=ConfidenceLevel.MEDIUM,
                rationale="Applicable policy/rule text found.",
                evidence_ids=[e.id for e in rules],
            )
        if contradictions:
            return ClaimVerdict(
                claim_id=claim.id,
                verdict=Verdict.CONTRADICTED,
                confidence=ConfidenceLevel.LOW,
                rationale="Conflicting policy text found; reconciliation required.",
                evidence_ids=[e.id for e in evs],
                limitations=["Defer to the most recent or most specific policy."],
            )
        return ClaimVerdict(
            claim_id=claim.id,
            verdict=Verdict.UNCLEAR,
            confidence=ConfidenceLevel.LOW,
            rationale="No applicable policy/rule text available.",
            evidence_ids=[e.id for e in evs],
        )

    # Interpretive claims: synthesize but expose assumptions.
    if claim.epistemic_type == EpistemicType.INTERPRETIVE:
        if any(_supports(e) or _partial(e) for e in evs):
            return ClaimVerdict(
                claim_id=claim.id,
                verdict=Verdict.PARTIALLY_SUPPORTED,
                confidence=ConfidenceLevel.LOW,
                rationale="Interpretive synthesis based on available evidence.",
                evidence_ids=[e.id for e in evs],
                assumptions=["Synthesis may reflect framing choices, not ground truth."],
            )
        return ClaimVerdict(
            claim_id=claim.id,
            verdict=Verdict.UNCLEAR,
            confidence=ConfidenceLevel.LOW,
            rationale="Insufficient evidence for an interpretive synthesis.",
            evidence_ids=[e.id for e in evs],
        )

    # DIRECT_FACT and UNKNOWN fall through here.
    contradictions = [e for e in evs if _contradicts(e)]
    if contradictions:
        return ClaimVerdict(
            claim_id=claim.id,
            verdict=Verdict.CONTRADICTED,
            confidence=ConfidenceLevel.LOW,
            rationale="Sources disagree.",
            evidence_ids=[e.id for e in evs],
        )

    supports = [e for e in evs if _supports(e)]
    partial = [e for e in evs if _partial(e)]

    if supports:
        return ClaimVerdict(
            claim_id=claim.id,
            verdict=Verdict.SUPPORTED,
            confidence=(
                ConfidenceLevel.HIGH
                if _highest_quality(supports)
                in (SourceQuality.AUTHORITATIVE, SourceQuality.PRIMARY)
                else ConfidenceLevel.MEDIUM
            ),
            rationale="Direct factual support found.",
            evidence_ids=[e.id for e in supports],
        )
    if partial:
        return ClaimVerdict(
            claim_id=claim.id,
            verdict=Verdict.PARTIALLY_SUPPORTED,
            confidence=ConfidenceLevel.LOW,
            rationale="Partial textual support; not authoritative.",
            evidence_ids=[e.id for e in partial],
        )
    return ClaimVerdict(
        claim_id=claim.id,
        verdict=Verdict.UNSUPPORTED,
        confidence=ConfidenceLevel.LOW,
        rationale="No supporting evidence found.",
        evidence_ids=[e.id for e in evs],
    )


def assign_verdicts(claims: list[Claim], evidence: list[Evidence]) -> list[ClaimVerdict]:
    return [assign_verdict(c, evidence) for c in claims]
