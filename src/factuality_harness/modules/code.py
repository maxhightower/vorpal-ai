"""CodeModule (stub).

Code claims would be verified by running tests or executing snippets in a
sandbox. The interface is declared here; execution is intentionally deferred.
"""

from __future__ import annotations

from ..domain.modules import ModuleSpec, ModuleStatus
from .base import BaseDomainModule


class CodeModule(BaseDomainModule):
    def __init__(self) -> None:
        super().__init__(
            ModuleSpec(
                name="code",
                description="Code-related claims (stub — needs sandboxed executor).",
                version="0.0.1",
                status=ModuleStatus.DRAFT,
                claim_types=["unit_test", "snippet_correctness", "type_check"],
                tool_policy={"NUMERICAL": ["python_executor"]},
                evidence_standards={
                    "snippet_correctness": "Must be verified by passing tests.",
                },
            )
        )
