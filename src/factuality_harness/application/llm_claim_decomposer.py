"""LLM-backed claim decomposer with strict JSON output and a hard fallback.

The decomposer prompts an LLM to produce a JSON list of atomic claims for a
question. The output is strictly parsed; any malformed response is rejected
and the request falls back to the rule-based decomposer. This ensures the
harness never silently ingests free-form LLM prose as ``Claim`` objects.

The harness deliberately keeps this thin — the LLM is only here to *propose*
claims; routing, evidence gathering, verdicts, and final-answer revision
remain the safety net. Garbage proposals produce UNSUPPORTED verdicts; they
do not become asserted facts.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..domain.claims import Claim, ClaimStatus
from ..domain.epistemic_types import EpistemicType
from ..infrastructure.llm.base import LLM, LLMMessage, LLMRequest
from .claim_decomposer import ClaimDecomposer, RuleBasedClaimDecomposer


_SYSTEM_PROMPT = """\
You decompose a user question into atomic, independently verifiable claims.

Rules:
1. Output ONLY a JSON object of the form: {"claims": [<claim>, ...]}.
2. Each <claim> object has fields: "text" (string, the atomic claim), and
   optionally "epistemic_type" (one of: DIRECT_FACT, NUMERICAL, LOGICAL,
   PROCEDURAL, CAUSAL, PREDICTIVE, OPTIMIZATION, INTERPRETIVE, SPECULATIVE,
   UNKNOWN).
3. Do not invent facts. Each claim must be derivable from the question text.
4. Prefer fewer, larger claims over over-splitting. Do not produce fragments
   that cannot be verified on their own.
5. Do not include any prose, code fences, comments, or extra fields.
"""


# Match the first balanced JSON object in a string. Handles fenced code blocks.
_JSON_RE = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    # Strip code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    # Otherwise grab the first JSON-looking object.
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class LLMClaimDecomposer:
    """Use an LLM to decompose questions; fall back on any failure."""

    def __init__(
        self,
        *,
        llm: LLM,
        fallback: ClaimDecomposer | None = None,
        max_claims: int = 12,
    ) -> None:
        self._llm = llm
        self._fallback = fallback or RuleBasedClaimDecomposer()
        self._max_claims = max_claims

    def decompose(self, question: str) -> list[Claim]:
        question = question.strip()
        if not question:
            return []

        try:
            response = self._llm.complete(
                LLMRequest(
                    messages=[
                        LLMMessage(role="system", content=_SYSTEM_PROMPT),
                        LLMMessage(role="user", content=question),
                    ],
                    temperature=0.0,
                )
            )
        except Exception:
            return self._fallback.decompose(question)

        parsed = _extract_json(response.text)
        if not isinstance(parsed, dict):
            return self._fallback.decompose(question)

        raw_claims = parsed.get("claims")
        if not isinstance(raw_claims, list) or not raw_claims:
            return self._fallback.decompose(question)

        claims: list[Claim] = []
        for entry in raw_claims[: self._max_claims]:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text")
            if not isinstance(text, str) or not text.strip():
                continue

            etype = EpistemicType.UNKNOWN
            raw_type = entry.get("epistemic_type")
            if isinstance(raw_type, str):
                try:
                    etype = EpistemicType(raw_type.strip().upper())
                except ValueError:
                    etype = EpistemicType.UNKNOWN

            claims.append(
                Claim(
                    text=text.strip(),
                    normalized_text=text.strip().lower(),
                    parent_question=question,
                    epistemic_type=etype,
                    status=(
                        ClaimStatus.CLASSIFIED
                        if etype != EpistemicType.UNKNOWN
                        else ClaimStatus.PROPOSED
                    ),
                )
            )

        if not claims:
            return self._fallback.decompose(question)
        return claims
