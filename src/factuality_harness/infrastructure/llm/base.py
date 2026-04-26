from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class LLMMessage(BaseModel):
    role: Role
    content: str


class LLMRequest(BaseModel):
    messages: list[LLMMessage]
    temperature: float = 0.0
    max_tokens: int | None = None
    metadata: dict = Field(default_factory=dict)


class LLMResponse(BaseModel):
    text: str
    raw: dict = Field(default_factory=dict)


class LLM(Protocol):
    """Vendor-neutral chat-completion interface.

    The harness must never let LLM output bypass verification, so this interface
    intentionally returns plain text only. Tool calls happen via the harness's
    own router, not via vendor-specific tool-use APIs.
    """

    name: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...
