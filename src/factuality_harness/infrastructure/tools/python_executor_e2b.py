"""E2B-backed Python executor.

Spins up a fresh E2B sandbox per call, runs the code, captures stdout
and stderr, kills the sandbox, and emits ``COMPUTATION`` evidence with
the same shape every other executor produces.

The ``e2b_code_interpreter`` package is imported lazily so the harness
doesn't require it at import time. Constructing the executor without
the package installed is fine; calling ``.run()`` returns a clean error
explaining what's missing.

Required environment::

    pip install e2b_code_interpreter
    export E2B_API_KEY=...

Expected context shape::

    request.context["code"] = "<python source>"
    request.context["timeout_s"] = 10.0   # per-execution timeout (default 10)
"""

from __future__ import annotations

import os
import re
from typing import Any

from ...domain.evidence import (
    FreshnessStatus,
    SourceQuality,
    SourceType,
    SupportStatus,
)
from .base import Tool, ToolRequest, ToolResult, build_evidence


_FENCED_PY = re.compile(r"```(?:python|py)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)


def _extract_code_from_claim(text: str) -> str | None:
    m = _FENCED_PY.search(text)
    return (m.group(1).strip() or None) if m else None


class E2BPythonExecutor:
    name = "python_executor"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_timeout_s: float = 10.0,
        sandbox_factory: Any | None = None,
    ) -> None:
        # Resolve the API key lazily — construction must succeed even
        # without one so the factory can build it ahead of any check.
        self._api_key = api_key
        self._timeout_s = default_timeout_s
        # Tests inject a fake sandbox factory; production reads the real SDK.
        self._sandbox_factory = sandbox_factory

    # ------------------------------------------------------------------

    def _resolve_sandbox_factory(self) -> Any | None:
        if self._sandbox_factory is not None:
            return self._sandbox_factory
        try:
            from e2b_code_interpreter import Sandbox  # type: ignore
        except ImportError:
            return None
        return Sandbox

    def run(self, request: ToolRequest) -> ToolResult:
        code = request.context.get("code")
        if not isinstance(code, str) or not code.strip():
            code = _extract_code_from_claim(request.claim.text)
        if not code:
            return ToolResult(
                succeeded=False,
                error=(
                    "E2BPythonExecutor requires context['code'] or fenced "
                    "```python``` in the claim."
                ),
            )

        sandbox_factory = self._resolve_sandbox_factory()
        if sandbox_factory is None:
            return ToolResult(
                succeeded=False,
                error=(
                    "e2b_code_interpreter is not installed. "
                    "`pip install e2b_code_interpreter` and set E2B_API_KEY."
                ),
            )

        api_key = self._api_key or os.environ.get("E2B_API_KEY")
        if not api_key:
            return ToolResult(
                succeeded=False,
                error="E2B_API_KEY is not set; cannot create a sandbox.",
            )

        timeout_s = float(request.context.get("timeout_s", self._timeout_s))

        try:
            sandbox = sandbox_factory(api_key=api_key, timeout=int(timeout_s))
        except Exception as e:
            return ToolResult(
                succeeded=False, error=f"Failed to launch E2B sandbox: {e}"
            )

        try:
            execution = sandbox.run_code(code, timeout=int(timeout_s))
        except Exception as e:
            return ToolResult(
                succeeded=False, error=f"E2B execution failed: {e}"
            )
        finally:
            # Best-effort cleanup; older SDKs use .close(), newer use .kill().
            for closer in ("kill", "close"):
                fn = getattr(sandbox, closer, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
                    break

        logs = getattr(execution, "logs", None)
        stdout_lines = list(getattr(logs, "stdout", []) or [])
        stderr_lines = list(getattr(logs, "stderr", []) or [])
        stdout = "\n".join(str(s) for s in stdout_lines)[-4000:]
        stderr = "\n".join(str(s) for s in stderr_lines)[-1000:]

        error_obj = getattr(execution, "error", None)
        if error_obj is not None:
            error_summary = (
                f"{getattr(error_obj, 'name', 'Error')}: "
                f"{getattr(error_obj, 'value', error_obj)}"
            )
            stderr = (stderr + "\n" + error_summary).strip()
            return_code = 1
            succeeded = False
        else:
            return_code = 0
            succeeded = True

        return ToolResult(
            succeeded=succeeded,
            evidence=[
                build_evidence(
                    claim_id=request.claim.id,
                    source_type=SourceType.COMPUTATION,
                    source_name="e2b_sandbox",
                    quote_or_result=(
                        f"return_code={return_code}\n"
                        f"--- stdout ---\n{stdout}"
                        + (f"\n--- stderr ---\n{stderr}" if stderr else "")
                    ),
                    normalized_result={
                        "return_code": return_code,
                        "stdout": stdout,
                        "stderr": stderr,
                        "timeout_s": timeout_s,
                        "backend": "e2b",
                    },
                    supports_claim=(
                        SupportStatus.SUPPORTS
                        if succeeded
                        else SupportStatus.INSUFFICIENT
                    ),
                    source_quality=(
                        SourceQuality.AUTHORITATIVE
                        if succeeded
                        else SourceQuality.UNKNOWN
                    ),
                    freshness=FreshnessStatus.NOT_TIME_SENSITIVE,
                    notes="Executed in an ephemeral E2B sandbox.",
                )
            ],
            error=None if succeeded else f"Process exited with {return_code}.",
        )


_: Tool = E2BPythonExecutor.__new__(E2BPythonExecutor)
