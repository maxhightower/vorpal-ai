"""Smoke-test every LLM-touching component against a real provider.

Why this exists: the unit tests for the LLM-backed components use mock
LLMs (``MockLLM``) and stubbed clients. They prove the *wiring* is
correct — the request shape, the response parser, the fallback paths —
but they never exercise a real LLM. This script does the missing half:
it sends each prompt to a real provider, prints what came back, and
flags whether the response matched the shape the parser expects.

Run::

    export ANTHROPIC_API_KEY=...   # preferred
    # or: export OPENAI_API_KEY=...
    python scripts/smoke_test_llm_apis.py

What it costs (rough order-of-magnitude per full run with claude-opus-4-7):

    ~2-5 cents in tokens.
    The code-executor probe additionally reserves a few seconds of
    Anthropic-hosted sandbox time (well within the free tier).

What you'll see, per component:

    [PASS] - real provider returned a response that parsed cleanly.
    [SOFT] - response came back but did NOT match the expected shape
             (the parser fell back). The raw response is printed so
             you can see *what* the LLM said vs. what we expected.
    [FAIL] - the call itself errored (network, auth, rate limit,
             bad request, etc.). Errors should be acted on.
    [SKIP] - this component requires a key that wasn't set.

Exit codes:

    0 if no FAIL results.
    1 if any FAIL result.

This script is NOT in the pytest suite — pytest must stay deterministic
and zero-cost. Run it manually after wiring a key, and re-run it after
prompt-tuning changes to catch regressions.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable

# Make src/ importable when the script is invoked directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))


from factuality_harness.application.contradiction_checker import (  # noqa: E402
    LLMContradictionDetector,
)
from factuality_harness.application.data_source_router import (  # noqa: E402
    LLMDataSourceRouter,
)
from factuality_harness.application.llm_claim_decomposer import (  # noqa: E402
    LLMClaimDecomposer,
)
from factuality_harness.application.tool_input_translator import (  # noqa: E402
    TOOL_INPUT_SPECS,
    LLMToolInputTranslator,
)
from factuality_harness.domain.catalog import (  # noqa: E402
    DataCatalog,
    DataCatalogEntry,
    SourceKind,
    TableSchema,
)
from factuality_harness.infrastructure.llm.base import (  # noqa: E402
    LLM,
    LLMMessage,
    LLMRequest,
)


_PASS = "PASS"
_SOFT = "SOFT"
_FAIL = "FAIL"
_SKIP = "SKIP"

_COLOR = {
    _PASS: "\033[92m",
    _SOFT: "\033[93m",
    _FAIL: "\033[91m",
    _SKIP: "\033[90m",
}
_RESET = "\033[0m"


@dataclass
class ProbeResult:
    name: str
    status: str
    summary: str = ""
    raw_response: str = ""
    expected: str = ""
    notes: list[str] = field(default_factory=list)
    usage: dict[str, Any] | None = None


def _truncate(s: str, limit: int = 600) -> str:
    s = (s or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit] + f" ... <truncated {len(s) - limit} chars>"


def _print_result(r: ProbeResult) -> None:
    color = _COLOR.get(r.status, "")
    print(f"\n{color}[{r.status}] {r.name}{_RESET}")
    if r.summary:
        print(f"  summary: {r.summary}")
    if r.expected:
        print(f"  expected: {r.expected}")
    if r.raw_response:
        print("  raw response:")
        for line in _truncate(r.raw_response).splitlines() or [""]:
            print(f"    {line}")
    if r.usage:
        print(f"  usage: {r.usage}")
    for note in r.notes:
        print(f"  note: {note}")


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def _build_anthropic_llm() -> LLM | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    from factuality_harness.infrastructure.llm.anthropic_adapter import (
        AnthropicAdapter,
    )

    return AnthropicAdapter()


def _build_openai_llm() -> LLM | None:
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    from factuality_harness.infrastructure.llm.openai_adapter import OpenAIAdapter

    return OpenAIAdapter()


def _select_llm() -> tuple[LLM | None, str]:
    """Anthropic preferred (matches the harness's defaults); OpenAI fallback."""
    llm = _build_anthropic_llm()
    if llm is not None:
        return llm, "anthropic"
    llm = _build_openai_llm()
    if llm is not None:
        return llm, "openai"
    return None, "none"


# ---------------------------------------------------------------------------
# Probes — adapter round-trips
# ---------------------------------------------------------------------------


def probe_anthropic_adapter() -> ProbeResult:
    name = "AnthropicAdapter.complete"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return ProbeResult(name=name, status=_SKIP, summary="ANTHROPIC_API_KEY not set")
    try:
        llm = _build_anthropic_llm()
        response = llm.complete(
            LLMRequest(
                messages=[
                    LLMMessage(role="system", content="Reply with exactly: ok"),
                    LLMMessage(role="user", content="ack"),
                ],
                max_tokens=16,
            )
        )
    except Exception as e:
        return ProbeResult(
            name=name, status=_FAIL, summary=f"{type(e).__name__}: {e}"
        )
    text = (response.text or "").strip().lower()
    status = _PASS if "ok" in text else _SOFT
    return ProbeResult(
        name=name,
        status=status,
        summary=f"got {len(response.text)} chars; model={response.raw.get('model')}",
        raw_response=response.text,
        expected="single-line response containing 'ok'",
        usage=response.raw.get("usage"),
    )


def probe_openai_adapter() -> ProbeResult:
    name = "OpenAIAdapter.complete"
    if not os.environ.get("OPENAI_API_KEY"):
        return ProbeResult(name=name, status=_SKIP, summary="OPENAI_API_KEY not set")
    try:
        llm = _build_openai_llm()
        response = llm.complete(
            LLMRequest(
                messages=[
                    LLMMessage(role="user", content="Reply with exactly the token: ok"),
                ],
                max_tokens=16,
            )
        )
    except Exception as e:
        return ProbeResult(
            name=name, status=_FAIL, summary=f"{type(e).__name__}: {e}"
        )
    text = (response.text or "").strip().lower()
    status = _PASS if "ok" in text else _SOFT
    return ProbeResult(
        name=name,
        status=status,
        summary=f"got {len(response.text)} chars; model={response.raw.get('model')}",
        raw_response=response.text,
        expected="response containing 'ok'",
        usage=response.raw.get("usage"),
    )


# ---------------------------------------------------------------------------
# Probe — Anthropic code execution tool
# ---------------------------------------------------------------------------


def probe_anthropic_code_executor() -> ProbeResult:
    name = "AnthropicCodeExecutor.run"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return ProbeResult(name=name, status=_SKIP, summary="ANTHROPIC_API_KEY not set")
    from factuality_harness.domain.claims import Claim
    from factuality_harness.domain.epistemic_types import EpistemicType
    from factuality_harness.infrastructure.tools.base import ToolRequest
    from factuality_harness.infrastructure.tools.python_executor_anthropic import (
        AnthropicCodeExecutor,
    )

    try:
        tool = AnthropicCodeExecutor()
        result = tool.run(
            ToolRequest(
                claim=Claim(
                    text="execute",
                    parent_question="x",
                    epistemic_type=EpistemicType.NUMERICAL,
                ),
                context={"code": "print(2 + 3)"},
            )
        )
    except Exception as e:
        return ProbeResult(
            name=name, status=_FAIL, summary=f"{type(e).__name__}: {e}"
        )

    if not result.evidence:
        return ProbeResult(
            name=name,
            status=_FAIL,
            summary=result.error or "no evidence emitted",
            expected="bash_code_execution_tool_result block parsed",
        )

    nr = result.evidence[0].normalized_result or {}
    stdout = (nr.get("stdout") or "").strip()
    return_code = nr.get("return_code")
    success = return_code == 0 and "5" in stdout
    return ProbeResult(
        name=name,
        status=_PASS if success else _SOFT,
        summary=f"return_code={return_code}, stdout~={stdout[:80]!r}",
        raw_response=json.dumps(nr, indent=2, default=str),
        expected="return_code=0 and stdout contains '5'",
        notes=(
            ["parsing handled bash_code_execution_tool_result correctly"]
            if success
            else [
                "real SDK response did not match parser expectations — "
                "see raw_response above; the parser may need adjustment"
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Probes — LLM-backed harness components
# ---------------------------------------------------------------------------


def probe_llm_claim_decomposer(llm: LLM) -> ProbeResult:
    name = "LLMClaimDecomposer.decompose"
    decomposer = LLMClaimDecomposer(llm=llm)
    question = (
        "What is the percentage increase from 100 to 125, and did that increase "
        "prove the campaign caused growth?"
    )

    # Capture the raw response by intercepting the LLM call once.
    captured: dict[str, str] = {}
    real_complete = llm.complete

    def _spy(req):
        resp = real_complete(req)
        captured.setdefault("text", resp.text)
        return resp

    llm.complete = _spy  # type: ignore[assignment]
    try:
        claims = decomposer.decompose(question)
    except Exception as e:
        return ProbeResult(name=name, status=_FAIL, summary=str(e))
    finally:
        llm.complete = real_complete  # type: ignore[assignment]

    raw = captured.get("text", "")
    # The fallback yields the rule-based decomposer's output: 2 claims for
    # the compound question. If we got 2+ claims AND the raw text contains
    # JSON, the LLM produced parseable output. If 2 claims but no JSON, we
    # fell back to the rule-based path.
    parsed_via_llm = "{" in raw and '"claims"' in raw
    status = _PASS if parsed_via_llm else _SOFT
    return ProbeResult(
        name=name,
        status=status,
        summary=(
            f"got {len(claims)} claim(s); "
            + ("parsed from real LLM JSON" if parsed_via_llm else "fell back to rule-based")
        ),
        raw_response=raw,
        expected='JSON object {"claims": [...]} with at least 2 entries',
        notes=(
            []
            if parsed_via_llm
            else [
                "LLM output was not strict JSON; the rule-based fallback "
                "produced the claims. Tune the system prompt and/or the "
                "JSON extractor."
            ]
        ),
    )


def probe_llm_tool_input_translator(llm: LLM) -> ProbeResult:
    name = "LLMToolInputTranslator.translate"
    translator = LLMToolInputTranslator(llm=llm)
    rows = [
        {"received_email": 1, "converted": 1, "loyalty": 0.4},
        {"received_email": 0, "converted": 0, "loyalty": -0.1},
        {"received_email": 1, "converted": 0, "loyalty": 0.0},
    ] * 20  # 60 rows so the causal_inference >=30 row check passes

    captured: dict[str, str] = {}
    real_complete = llm.complete

    def _spy(req):
        resp = real_complete(req)
        captured.setdefault("text", resp.text)
        return resp

    llm.complete = _spy  # type: ignore[assignment]
    try:
        out = translator.translate(
            question="Did the email cause more conversions?",
            tool_specs=[TOOL_INPUT_SPECS["causal_inference"]],
            existing_context={"data": {"experiment": rows}},
        )
    except Exception as e:
        return ProbeResult(name=name, status=_FAIL, summary=str(e))
    finally:
        llm.complete = real_complete  # type: ignore[assignment]

    raw = captured.get("text", "")
    if "causal_inference" in out:
        cols = list(rows[0].keys())
        treatment = out["causal_inference"].get("treatment")
        outcome = out["causal_inference"].get("outcome")
        common = out["causal_inference"].get("common_causes") or []
        sane = (
            treatment in cols
            and outcome in cols
            and all(c in cols for c in common)
            and treatment != outcome
        )
        return ProbeResult(
            name=name,
            status=_PASS if sane else _SOFT,
            summary=(
                f"emitted causal_inference payload "
                f"(treatment={treatment!r}, outcome={outcome!r})"
            ),
            raw_response=raw,
            expected=(
                "JSON object {causal_inference: {data, treatment, outcome, "
                "common_causes}} with column names from the supplied data"
            ),
            notes=(
                []
                if sane
                else [
                    "Translator picked column names that don't appear in the "
                    "data. Tighten the system prompt."
                ]
            ),
        )
    return ProbeResult(
        name=name,
        status=_SOFT,
        summary="no causal_inference payload emitted",
        raw_response=raw,
        expected="JSON with a 'causal_inference' key",
        notes=[
            "Real LLM response did not produce a usable payload; either it "
            "didn't follow the strict-JSON instruction or it omitted the tool. "
            "Tune the prompt or the validator."
        ],
    )


_NLI_CASES = [
    {
        "label": "expected CONTRADICT",
        "a": "Refunds must be issued within 30 days of the request.",
        "b": "Refunds are processed once per fiscal quarter.",
        "expect_contradict": True,
    },
    {
        "label": "expected NOT CONTRADICT (entail)",
        "a": "All meetings are at 9am.",
        "b": "The standup meeting is at 9am.",
        "expect_contradict": False,
    },
    {
        "label": "expected NOT CONTRADICT (neutral)",
        "a": "The sky appears blue on clear days.",
        "b": "Roses can be red, white, or yellow.",
        "expect_contradict": False,
    },
]


def probe_llm_contradiction_detector(llm: LLM) -> ProbeResult:
    name = "LLMContradictionDetector.is_contradicting"
    detector = LLMContradictionDetector(llm=llm)
    captured: list[str] = []
    real_complete = llm.complete

    def _spy(req):
        resp = real_complete(req)
        captured.append(resp.text)
        return resp

    llm.complete = _spy  # type: ignore[assignment]

    correct = 0
    case_lines: list[str] = []
    try:
        for case in _NLI_CASES:
            try:
                got = detector.is_contradicting(case["a"], case["b"])
            except Exception as e:
                return ProbeResult(name=name, status=_FAIL, summary=str(e))
            ok = got == case["expect_contradict"]
            correct += int(ok)
            case_lines.append(
                f"  - {case['label']}: returned {got}  ({'OK' if ok else 'MISS'})"
            )
    finally:
        llm.complete = real_complete  # type: ignore[assignment]

    status = (
        _PASS
        if correct == len(_NLI_CASES)
        else _SOFT
        if correct >= 1
        else _FAIL
    )
    return ProbeResult(
        name=name,
        status=status,
        summary=f"{correct}/{len(_NLI_CASES)} cases correct",
        raw_response="\n".join(case_lines)
        + "\n\nlast raw label: "
        + (captured[-1] if captured else "<none>"),
        expected=(
            "all 3 cases match expectation; raw labels should be one of "
            "CONTRADICT/ENTAIL/NEUTRAL"
        ),
        notes=(
            ["all 3 NLI judgments matched"]
            if status == _PASS
            else [
                "Real Claude got at least one NLI case wrong. Likely tuning "
                "needed: more emphatic prompt, examples, or reduce 'be "
                "conservative' bias."
            ]
        ),
    )


def probe_llm_data_source_router(llm: LLM) -> ProbeResult:
    name = "LLMDataSourceRouter.route"
    catalog = DataCatalog()
    catalog.register(
        DataCatalogEntry(
            source_id="financials",
            kind=SourceKind.WAREHOUSE,
            description="Quarterly revenue and gross margin per company.",
            tables=[
                TableSchema(
                    name="quarterly_revenue",
                    columns={"company": "VARCHAR", "quarter": "VARCHAR", "revenue": "DOUBLE"},
                )
            ],
            keywords=["revenue", "quarterly", "earnings"],
        )
    )
    catalog.register(
        DataCatalogEntry(
            source_id="weather",
            kind=SourceKind.REST_API,
            description="Daily temperature and precipitation by station.",
            keywords=["weather", "temperature", "precipitation"],
        )
    )

    captured: dict[str, str] = {}
    real_complete = llm.complete

    def _spy(req):
        resp = real_complete(req)
        captured.setdefault("text", resp.text)
        return resp

    llm.complete = _spy  # type: ignore[assignment]
    router = LLMDataSourceRouter(llm=llm)
    try:
        out = router.route(
            question="What was last quarter's revenue?",
            claims=[],
            catalog=catalog,
        )
    except Exception as e:
        return ProbeResult(name=name, status=_FAIL, summary=str(e))
    finally:
        llm.complete = real_complete  # type: ignore[assignment]

    raw = captured.get("text", "")
    selected = [e.source_id for e in out]
    if selected == ["financials"]:
        status = _PASS
    elif "financials" in selected:
        status = _SOFT
    else:
        status = _SOFT
    return ProbeResult(
        name=name,
        status=status,
        summary=f"selected={selected}",
        raw_response=raw,
        expected='JSON {"selected": ["financials"]}',
        notes=(
            ["picked the right source"]
            if selected == ["financials"]
            else [
                "Router returned the wrong (or no) source. The prompt may "
                "need a tighter instruction to select fewer rather than more."
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 78)
    print("LLM smoke test")
    print("=" * 78)
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    print(f"  ANTHROPIC_API_KEY: {'set' if has_anthropic else 'absent'}")
    print(f"  OPENAI_API_KEY:    {'set' if has_openai else 'absent'}")

    if not (has_anthropic or has_openai):
        print()
        print(
            textwrap.dedent(
                """\
                No LLM API key is set.

                Set ANTHROPIC_API_KEY (preferred — defaults match the harness)
                or OPENAI_API_KEY and re-run. Each probe will exit cleanly
                with status SKIP for any component that requires the missing key.

                Estimated cost when run with a key: a few cents per full pass.
                """
            )
        )
        return 0

    results: list[ProbeResult] = []

    # Adapter round-trips.
    results.append(probe_anthropic_adapter())
    results.append(probe_openai_adapter())

    # Anthropic-only: code-execution server tool.
    results.append(probe_anthropic_code_executor())

    # Pick a single LLM for the harness components — Anthropic preferred.
    llm, provider = _select_llm()
    if llm is None:
        results.append(
            ProbeResult(
                name="<harness components>",
                status=_SKIP,
                summary="no LLM available",
            )
        )
    else:
        print(f"\n  using provider {provider!r} for harness components\n")
        results.append(probe_llm_claim_decomposer(llm))
        results.append(probe_llm_tool_input_translator(llm))
        results.append(probe_llm_contradiction_detector(llm))
        results.append(probe_llm_data_source_router(llm))

    for r in results:
        _print_result(r)

    print("\n" + "=" * 78)
    counts = {s: 0 for s in (_PASS, _SOFT, _FAIL, _SKIP)}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print(
        f"Summary: {counts[_PASS]} PASS, {counts[_SOFT]} SOFT, "
        f"{counts[_FAIL]} FAIL, {counts[_SKIP]} SKIP"
    )
    print("=" * 78)
    return 1 if counts[_FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
