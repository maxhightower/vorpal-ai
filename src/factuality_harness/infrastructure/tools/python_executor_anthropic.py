"""Anthropic-backed Python executor.

Uses Claude's server-side ``code_execution_20260120`` tool. The harness
sends Claude the code with strict instructions to run it as-is via the
tool, then parses the resulting ``bash_code_execution_tool_result`` block
back into the same evidence shape every other executor produces.

Trust model and tradeoffs
 - Pro: no second vendor; reuses the ``anthropic`` SDK already wired in.
 - Pro: Anthropic-hosted sandbox — no infrastructure to run.
 - Pro: free tier covers most prototype workloads.
 - Caveat: there is an LLM in the middle. Claude can in principle decline
   or modify the request; we constrain it with a system prompt and parse
   *only* the tool-result blocks. If Claude doesn't call the tool, this
   executor returns a clean error rather than fabricating output.

Expected context shape (same as every other executor)::

    request.context["code"] = "<python source>"
    request.context["timeout_s"] = 5.0          # advisory; passed to Claude
"""

from __future__ import annotations

import os
import re
from typing import Any

import anthropic

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


_SYSTEM_PROMPT = """\
You are a deterministic code-execution backend.

When the user provides a Python code block, IMMEDIATELY call the
code_execution tool to execute the code EXACTLY AS WRITTEN. You must:

1. Use the tool. Do not answer in prose.
2. Run the supplied code unchanged. Do not edit, refactor, or wrap it.
3. Do not add imports or print statements that the user did not write.
4. Do not summarize, interpret, or paraphrase the output.

If the code raises, return the failure exactly as the tool reports it.
"""


class AnthropicCodeExecutor:
    name = "python_executor"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "claude-opus-4-7",
        max_tokens: int = 4096,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        if client is not None:
            self._client = client
        else:
            self._client = anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
            )

    def run(self, request: ToolRequest) -> ToolResult:
        code = request.context.get("code")
        if not isinstance(code, str) or not code.strip():
            code = _extract_code_from_claim(request.claim.text)
        if not code:
            return ToolResult(
                succeeded=False,
                error=(
                    "AnthropicCodeExecutor requires context['code'] or fenced "
                    "```python``` in the claim."
                ),
            )

        user_message = (
            "Execute this Python code and return its output:\n\n"
            f"```python\n{code}\n```"
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                tools=[
                    {
                        "type": "code_execution_20260120",
                        "name": "code_execution",
                    }
                ],
            )
        except Exception as e:
            return ToolResult(
                succeeded=False, error=f"Anthropic API call failed: {e}"
            )

        # Parse the response for the bash_code_execution_tool_result block.
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type != "bash_code_execution_tool_result":
                continue
            inner = getattr(block, "content", None)
            inner_type = getattr(inner, "type", None)
            if inner_type != "bash_code_execution_result":
                # Tool errored out at the platform level (sandbox unavailable, etc.)
                error_code = getattr(inner, "error_code", "unknown")
                return ToolResult(
                    succeeded=False,
                    error=f"Anthropic code_execution platform error: {error_code}",
                )
            return_code = int(getattr(inner, "return_code", -1))
            stdout = (str(getattr(inner, "stdout", "") or ""))[-4000:]
            stderr = (str(getattr(inner, "stderr", "") or ""))[-1000:]
            succeeded = return_code == 0

            return ToolResult(
                succeeded=succeeded,
                evidence=[
                    build_evidence(
                        claim_id=request.claim.id,
                        source_type=SourceType.COMPUTATION,
                        source_name="anthropic_code_execution",
                        quote_or_result=(
                            f"return_code={return_code}\n"
                            f"--- stdout ---\n{stdout}"
                            + (f"\n--- stderr ---\n{stderr}" if stderr else "")
                        ),
                        normalized_result={
                            "return_code": return_code,
                            "stdout": stdout,
                            "stderr": stderr,
                            "model": getattr(response, "model", self._model),
                            "backend": "anthropic_code_execution",
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
                        notes=(
                            "Executed in Anthropic-hosted sandbox via the "
                            "code_execution tool."
                        ),
                    )
                ],
                error=None if succeeded else f"Process exited with {return_code}.",
            )

        # Claude responded with prose instead of using the tool.
        return ToolResult(
            succeeded=False,
            error=(
                "Anthropic model did not invoke the code_execution tool. "
                "Cannot return execution result."
            ),
        )


_: Tool = AnthropicCodeExecutor.__new__(AnthropicCodeExecutor)
