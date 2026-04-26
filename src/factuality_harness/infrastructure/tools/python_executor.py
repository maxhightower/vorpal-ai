"""Stub Python executor tool.

A real implementation would run sandboxed code (e.g. via a subprocess with
resource limits, or a remote sandbox). For the MVP we expose the interface and
return INSUFFICIENT evidence so callers can replace it without surprise.
"""

from __future__ import annotations

from .base import Tool, ToolRequest, ToolResult


class PythonExecutorStub:
    name = "python_executor"

    def run(self, request: ToolRequest) -> ToolResult:
        return ToolResult(
            succeeded=False,
            error="PythonExecutorStub is not implemented. Provide a sandboxed executor.",
        )


_: Tool = PythonExecutorStub()
