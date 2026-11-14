"""Ports for the system under test (SUT), its environment and the user simulator.

A SUT never sees a raw provider. It gets a :class:`ChatModel` already bound to its role's model,
token limit and call context, so SUT code cannot bypass metering or mislabel its calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import Field

from assayer.domain.base import Model
from assayer.domain.fixtures import ConversationFixture, Persona, Scenario, TaskFixture
from assayer.domain.messages import CompletionResponse, Message, ToolSpec
from assayer.domain.outputs import TaskOutput, ToolEvent


class ChatModel(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def calls_made(self) -> int: ...

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        response_schema: dict[str, Any] | None = None,
        turn: int | None = None,
    ) -> CompletionResponse: ...


class TaskRunner(Protocol):
    async def run(self, fixture: TaskFixture, llm: ChatModel) -> TaskOutput: ...


class ToolError(Exception):
    """A tool rejected its input. Reported back to the agent, not raised to the harness."""


class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...

    def invoke(self, arguments: dict[str, Any]) -> Any: ...


class Environment(Protocol):
    """Per-attempt world the agent acts on (e.g. an in-memory HR system)."""

    @property
    def tools(self) -> Sequence[Tool]: ...

    def snapshot(self) -> dict[str, Any]: ...


class EnvironmentFactory(Protocol):
    def create(self, fixture: ConversationFixture) -> Environment: ...


class AgentTurn(Model):
    messages: tuple[Message, ...]
    tool_events: tuple[ToolEvent, ...] = ()
    ended_conversation: bool = False
    error: str | None = None  # agent misbehaviour, e.g. exceeded its tool-call budget


class ConversationalAgent(Protocol):
    async def respond(
        self, history: Sequence[Message], env: Environment, llm: ChatModel, *, turn: int
    ) -> AgentTurn: ...


class SimTurn(Model):
    message: str = ""
    done: bool = False
    reason: str = Field(default="", description="goal_met | gave_up | continuing")


class UserSimulator(Protocol):
    async def next_turn(
        self,
        persona: Persona,
        scenario: Scenario,
        history: Sequence[Message],
        llm: ChatModel,
        *,
        turn: int,
    ) -> SimTurn: ...
