"""Provider-neutral chat messages, tool calls and completion request/response."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from assayer.domain.base import Model

Role = Literal["system", "user", "assistant", "tool"]


class Purpose(StrEnum):
    """Why a model call is made. Drives budgeting, fake-script matching and reporting."""

    AGENT = "agent"
    SIMULATOR = "simulator"
    JUDGE = "judge"


class CallContext(Model):
    """Who is calling and why. Real providers ignore it; metering and fakes use it."""

    purpose: Purpose
    run_id: str | None = None
    case_id: str | None = None
    attempt_id: int | None = None
    turn: int | None = None
    judgement_key: str | None = None


class ToolCall(Model):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolSpec(Model):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema


class Message(Model):
    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str | None, tool_calls: tuple[ToolCall, ...] = ()) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> Message:
        return cls(role="tool", content=content, tool_call_id=tool_call_id)


class Usage(Model):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class CompletionRequest(Model):
    model: str
    messages: tuple[Message, ...]
    context: CallContext
    tools: tuple[ToolSpec, ...] = ()
    response_schema: dict[str, Any] | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float = 0.0
    seed: int | None = None


class CompletionResponse(Model):
    message: Message
    model: str
    usage: Usage | None = None
    latency_ms: float = 0.0
