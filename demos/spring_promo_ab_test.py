"""End-to-end demo: a real campaign-lift question against a real A/B test.

Scenario (synthetic but fully specified — not real customer data):

  Spring 2026 promotional-email campaign, run as a randomized 50/50 A/B test
  against the company's prior 4-week behavioural baseline. The control group
  received no email; the treatment group received the promotional email. The
  primary metric is 7-day purchase conversion.

      Treatment:  n = 50,000   conversions = 1,250   (2.500%)
      Control:    n = 50,000   conversions = 1,000   (2.000%)
      Window:     2026-03-01 .. 2026-03-08
      Design:     simple randomization, intent-to-treat

  Question we put through the harness:
      "By how much did purchase conversion change from 2.00% to 2.50%, and
       did the Spring 2026 promotional email cause that increase?"

The harness will:
  1. decompose into a numerical claim and a causal claim,
  2. compute the lift via the calculator,
  3. run a real two-proportion z-test via ABTestCausalTool,
  4. assign verdicts based on the actual statistical evidence.
"""

from __future__ import annotations

import json

from factuality_harness.application.pipeline import (
    FactualityPipeline,
    PipelineRequest,
)
from factuality_harness.application.evidence_builder import EvidenceBuilder
from factuality_harness.infrastructure.storage.repository import (
    InMemoryAuditRepository,
)
from factuality_harness.infrastructure.tools.ab_test import ABTestCausalTool
from factuality_harness.infrastructure.tools.calculator import CalculatorTool
from factuality_harness.infrastructure.tools.optimizer_stub import OptimizerStub
from factuality_harness.infrastructure.tools.rule_engine import RuleEngineTool
from factuality_harness.infrastructure.tools.theorem_prover_stub import (
    TheoremProverStub,
)


EXPERIMENT = {
    "name": "spring_2026_promo_email",
    "metric": "purchase_conversion_7d",
    "design": "randomized A/B test, 50/50, intent-to-treat",
    "start": "2026-03-01",
    "end": "2026-03-08",
    "alpha": 0.05,
    "treatment": {"n": 50_000, "conversions": 1_250},
    "control":   {"n": 50_000, "conversions": 1_000},
}


def build_pipeline() -> FactualityPipeline:
    # Replace the causal_model stub with the real two-proportion z-test tool.
    tools = {
        "calculator": CalculatorTool(),
        "rule_engine": RuleEngineTool(),
        "theorem_prover": TheoremProverStub(),
        "causal_model": ABTestCausalTool(),
        "optimizer": OptimizerStub(),
    }
    repo = InMemoryAuditRepository()
    pipeline = FactualityPipeline(audit_repo=repo)
    pipeline.evidence_builder = EvidenceBuilder(
        tools=tools, retriever=pipeline.retriever
    )
    return pipeline


def banner(s: str) -> None:
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def main() -> None:
    pipeline = build_pipeline()

    question = (
        "By how much did purchase conversion change from 2.00 to 2.50, "
        "and did the Spring 2026 promotional email cause that increase?"
    )

    banner("INPUT")
    print("Question:", question)
    print("Experiment payload:")
    print(json.dumps(EXPERIMENT, indent=2))

    request = PipelineRequest(
        question=question,
        extra_context={"experiment": EXPERIMENT},
    )
    final = pipeline.run(request)
    trace = pipeline.audit_repo.get(final.audit_id)

    banner("DECOMPOSED + CLASSIFIED CLAIMS")
    for c in trace.decomposed_claims:
        print(f"- [{c.epistemic_type.value}] {c.text!r}")

    banner("TOOL CALLS")
    for tc in trace.tools_called:
        print(f"- {tc.tool_name}: succeeded={tc.succeeded} outputs={tc.outputs}"
              + (f" error={tc.error}" if tc.error else ""))

    banner("EVIDENCE")
    for e in trace.retrieved_evidence:
        print(f"- {e.source_type.value}/{e.source_quality.value} "
              f"({e.supports_claim.value}) — {e.source_name}")
        print(f"    {e.quote_or_result}")
        if e.normalized_result:
            print("    normalized_result:")
            print("      " + json.dumps(e.normalized_result, indent=6, default=str)
                  .replace("\n", "\n      "))

    banner("VERDICTS")
    for v in trace.verdicts:
        claim = next(c for c in trace.decomposed_claims if c.id == v.claim_id)
        print(f"- [{claim.epistemic_type.value}] {claim.text!r}")
        print(f"    verdict: {v.verdict.value}  confidence: {v.confidence.value}")
        print(f"    rationale: {v.rationale}")
        if v.limitations:
            print(f"    limitations: {v.limitations}")

    banner("FINAL ANSWER")
    print(final.answer)
    print()
    print("audit_id:", final.audit_id)
    print("confidence_summary:", final.confidence_summary)
    print("unsupported_or_uncertain_claims:", final.unsupported_or_uncertain_claims)


if __name__ == "__main__":
    main()
