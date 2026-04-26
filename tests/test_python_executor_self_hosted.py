from __future__ import annotations

from factuality_harness.domain.claims import Claim
from factuality_harness.domain.epistemic_types import EpistemicType
from factuality_harness.infrastructure.tools.base import ToolRequest
from factuality_harness.infrastructure.tools.python_executor_self_hosted import (
    SelfHostedPythonExecutorStub,
)


def test_stub_returns_clear_error():
    tool = SelfHostedPythonExecutorStub()
    result = tool.run(
        ToolRequest(
            claim=Claim(
                text="x",
                parent_question="x",
                epistemic_type=EpistemicType.NUMERICAL,
            ),
            context={"code": "print(1)"},
        )
    )
    assert not result.succeeded
    assert "not implemented" in (result.error or "").lower()
