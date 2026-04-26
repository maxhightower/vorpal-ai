"""Anthropic adapter test using a stubbed Anthropic client.

Avoids any real API call: we monkey-patch ``client.messages.create`` to return
a hand-crafted Message with the response text we want to verify.
"""

from __future__ import annotations

from types import SimpleNamespace

import anthropic

from factuality_harness.infrastructure.llm.anthropic_adapter import AnthropicAdapter
from factuality_harness.infrastructure.llm.base import LLMMessage, LLMRequest


def _stub_message(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="msg_test",
        model="claude-opus-4-7",
        stop_reason="end_turn",
        usage=SimpleNamespace(
            model_dump=lambda: {"input_tokens": 10, "output_tokens": 4}
        ),
        content=[SimpleNamespace(type="text", text=text)],
    )


class _StubMessages:
    def __init__(self) -> None:
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _stub_message("ok")


class _StubClient:
    def __init__(self) -> None:
        self.messages = _StubMessages()


def test_passes_messages_and_system_separately():
    client = _StubClient()
    adapter = AnthropicAdapter(client=client, model="claude-opus-4-7")

    response = adapter.complete(
        LLMRequest(
            messages=[
                LLMMessage(role="system", content="You are precise."),
                LLMMessage(role="user", content="Hello"),
            ],
            max_tokens=100,
        )
    )

    assert response.text == "ok"
    assert client.messages.last_kwargs["system"] == "You are precise."
    assert client.messages.last_kwargs["messages"] == [{"role": "user", "content": "Hello"}]
    assert client.messages.last_kwargs["model"] == "claude-opus-4-7"
    assert client.messages.last_kwargs["max_tokens"] == 100


def test_cache_system_wraps_in_block():
    client = _StubClient()
    adapter = AnthropicAdapter(client=client, cache_system=True)
    adapter.complete(
        LLMRequest(
            messages=[
                LLMMessage(role="system", content="frozen"),
                LLMMessage(role="user", content="?"),
            ]
        )
    )
    sys_field = client.messages.last_kwargs["system"]
    assert isinstance(sys_field, list)
    assert sys_field[0]["cache_control"] == {"type": "ephemeral"}


def test_protocol_compatibility():
    # Confirm AnthropicAdapter satisfies the LLM protocol.
    from factuality_harness.infrastructure.llm.base import LLM

    adapter: LLM = AnthropicAdapter.__new__(AnthropicAdapter)
    assert adapter.name == "anthropic"
    # Sanity: the official anthropic SDK is the dependency the adapter uses.
    assert hasattr(anthropic, "Anthropic")
