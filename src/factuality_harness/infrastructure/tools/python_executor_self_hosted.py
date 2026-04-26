"""Self-hosted Python executor — stub.

For organizations that need to keep code execution on infrastructure they
operate themselves (data-residency, network-isolation, or compliance
constraints). This stub is the placeholder behind the same ``Tool``
protocol used by ``LocalSubprocessPythonExecutor``,
``AnthropicCodeExecutor``, and ``E2BPythonExecutor``.

What you'd implement here, in roughly increasing order of effort:

1. **Docker + seccomp.** Spawn a container per call from a hardened
   base image, mount nothing, drop all capabilities, kill on timeout.
   Cheapest to set up, weakest isolation; not appropriate for
   adversarial input.
2. **gVisor.** Same shape as (1) but with the ``runsc`` runtime in
   place of ``runc``. Strong syscall isolation; 95th-percentile latency
   higher than runc by ~50-100ms.
3. **Firecracker microVM.** What AWS Lambda and E2B run on under the
   hood. Strong isolation (full VM boundary), millisecond cold starts,
   highest operational overhead — image building, snapshot management,
   networking config.

Whichever you pick, the implementation should:

 - read ``request.context["code"]``,
 - launch an isolated environment per call,
 - enforce a timeout (``request.context.get("timeout_s", default)``),
 - capture stdout/stderr/return code,
 - emit ``COMPUTATION`` evidence in the same shape as the other
   executors so the verdict calibrator sees no difference.

This stub returns a clean error so the rest of the harness keeps
working (the verdict calibrator handles the missing evidence as
``UNCLEAR``/``UNSUPPORTED``, never fabricated).
"""

from __future__ import annotations

from .base import Tool, ToolRequest, ToolResult


class SelfHostedPythonExecutorStub:
    name = "python_executor"

    def run(self, request: ToolRequest) -> ToolResult:
        return ToolResult(
            succeeded=False,
            error=(
                "SelfHostedPythonExecutorStub: not implemented. Wire your "
                "Docker / gVisor / Firecracker setup behind this protocol "
                "and return COMPUTATION evidence the same way "
                "LocalSubprocessPythonExecutor does."
            ),
        )


_: Tool = SelfHostedPythonExecutorStub()
