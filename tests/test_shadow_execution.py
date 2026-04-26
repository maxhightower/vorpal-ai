"""Shadow execution: SHADOW modules observe but never influence the final
answer. Tests cover (a) verdicts get recorded, (b) the final answer is
unchanged, (c) shadow-module exceptions don't crash the pipeline."""

from __future__ import annotations

from factuality_harness.application.module_registry import ModuleRegistry
from factuality_harness.application.pipeline import (
    FactualityPipeline,
    PipelineRequest,
)
from factuality_harness.domain.confidence import ConfidenceLevel
from factuality_harness.domain.modules import ModuleSpec, ModuleStatus
from factuality_harness.domain.verdicts import ClaimVerdict, Verdict
from factuality_harness.infrastructure.storage.repository import (
    InMemoryAuditRepository,
)
from factuality_harness.modules.base import BaseDomainModule


def _make_pipeline(modules) -> FactualityPipeline:
    registry = ModuleRegistry(modules=modules)
    return FactualityPipeline(
        audit_repo=InMemoryAuditRepository(),
        module_registry=registry,
    )


class _AlwaysAgreeShadow(BaseDomainModule):
    """SHADOW module that mirrors whatever the active path concluded."""

    def __init__(self) -> None:
        super().__init__(
            ModuleSpec(
                name="shadow_mirror",
                description="Always proposes a SUPPORTED verdict.",
                version="0.0.1",
                status=ModuleStatus.SHADOW,
            )
        )

    def validate_evidence(self, claim, evidence):
        return ClaimVerdict(
            claim_id=claim.id,
            verdict=Verdict.SUPPORTED,
            confidence=ConfidenceLevel.LOW,
            rationale="shadow says supported",
        )


class _CrashingShadow(BaseDomainModule):
    def __init__(self) -> None:
        super().__init__(
            ModuleSpec(
                name="shadow_crash",
                description="Raises whenever asked.",
                version="0.0.1",
                status=ModuleStatus.SHADOW,
            )
        )

    def validate_evidence(self, claim, evidence):
        raise RuntimeError("shadow exploded")


# ---------------------------------------------------------------------------


def test_no_shadow_modules_leaves_audit_untouched():
    pipeline = _make_pipeline([])
    final = pipeline.run(
        PipelineRequest(question="What is the percentage increase from 100 to 125?")
    )
    trace = pipeline.audit_repo.get(final.audit_id)
    assert trace is not None
    assert trace.shadow_verdicts == []


def test_shadow_module_verdicts_recorded_to_audit_trace():
    pipeline = _make_pipeline([_AlwaysAgreeShadow()])
    final = pipeline.run(
        PipelineRequest(question="What is the percentage increase from 100 to 125?")
    )
    trace = pipeline.audit_repo.get(final.audit_id)
    assert trace is not None
    assert trace.shadow_verdicts, "expected shadow verdicts to be recorded"
    sv = trace.shadow_verdicts[0]
    assert sv.module_name == "shadow_mirror"
    assert sv.proposed_verdict == "SUPPORTED"


def test_shadow_module_does_not_change_final_answer():
    """Verify that running with a shadow module produces the same final
    answer as running without it."""
    baseline = _make_pipeline([])
    shadow = _make_pipeline([_AlwaysAgreeShadow()])
    request = PipelineRequest(
        question="What is the percentage increase from 100 to 125?"
    )
    base_result = baseline.run(request)
    shadow_result = shadow.run(request)
    assert base_result.answer == shadow_result.answer
    assert base_result.confidence_summary == shadow_result.confidence_summary


def test_shadow_module_exception_does_not_crash_pipeline():
    pipeline = _make_pipeline([_CrashingShadow()])
    final = pipeline.run(
        PipelineRequest(question="What is the percentage increase from 100 to 125?")
    )
    trace = pipeline.audit_repo.get(final.audit_id)
    assert trace is not None
    # The crashing shadow's outputs are dropped; the run still completes.
    assert trace.shadow_verdicts == []


def test_active_module_does_not_get_shadow_recorded():
    """Modules in ACTIVE status should not have their verdicts recorded
    via the shadow path — only SHADOW modules do."""

    class _ActiveModule(BaseDomainModule):
        def __init__(self) -> None:
            super().__init__(
                ModuleSpec(
                    name="active_one",
                    description="x",
                    version="0.0.1",
                    status=ModuleStatus.ACTIVE,
                )
            )

        def validate_evidence(self, claim, evidence):
            return ClaimVerdict(
                claim_id=claim.id,
                verdict=Verdict.SUPPORTED,
                confidence=ConfidenceLevel.LOW,
                rationale="x",
            )

    pipeline = _make_pipeline([_ActiveModule()])
    final = pipeline.run(
        PipelineRequest(question="What is the percentage increase from 100 to 125?")
    )
    trace = pipeline.audit_repo.get(final.audit_id)
    assert trace.shadow_verdicts == []


def test_shadow_agreement_flag_correct():
    """When the shadow proposes the same verdict as the active path on a
    NUMERICAL claim, ``agrees_with_active`` should be True. Here the
    active will produce COMPUTED and the shadow proposes SUPPORTED — so
    they should *not* agree."""
    pipeline = _make_pipeline([_AlwaysAgreeShadow()])
    final = pipeline.run(
        PipelineRequest(question="What is the percentage increase from 100 to 125?")
    )
    trace = pipeline.audit_repo.get(final.audit_id)
    sv = trace.shadow_verdicts[0]
    # Numerical claim with calculator evidence -> COMPUTED in the active path.
    assert sv.agrees_with_active is False
