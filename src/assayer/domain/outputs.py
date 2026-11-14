"""What a system under test produces: a task output or a conversation transcript."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter

from assayer.domain.base import Model
from assayer.domain.messages import Message, ToolCall


class TaskOutput(Model):
    """Output of a single-step task.

    ``agent_error`` is set when the *agent* misbehaved (e.g. its output did not parse). That is
    a graded failure, not a harness error (ADR-0008).
    """

    kind: Literal["task"] = "task"
    raw: str
    parsed: dict[str, Any] | None = None
    agent_error: str | None = None


class TerminationReason(StrEnum):
    GOAL_MET = "goal_met"
    USER_GAVE_UP = "user_gave_up"
    AGENT_ENDED = "agent_ended"
    TURN_LIMIT = "turn_limit"
    AGENT_TOOL_LOOP = "agent_tool_loop"
    SIMULATOR_LOOP = "simulator_loop"


class ToolEvent(Model):
    turn: int
    call: ToolCall
    result: Any = None
    error: str | None = None


class Transcript(Model):
    """Full record of a conversation, including tool activity and the environment's end state."""

    kind: Literal["conversation"] = "conversation"
    messages: tuple[Message, ...]
    tool_events: tuple[ToolEvent, ...] = ()
    final_state: dict[str, Any] = Field(default_factory=dict)
    termination: TerminationReason
    turns: int = 0
    agent_error: str | None = None

    def render(self) -> str:
        """Plain-text rendering used as the judge's evidence and for quote verification."""
        lines: list[str] = []
        for message in self.messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                lines.append(f"[tool result] {message.content}")
                continue
            speaker = "USER" if message.role == "user" else "AGENT"
            if message.content:
                lines.append(f"{speaker}: {message.content}")
            lines.extend(
                f"[tool call] {call.name}({call.arguments})" for call in message.tool_calls
            )
        return "\n".join(lines)


AgentOutput = TaskOutput | Transcript

_OUTPUT: TypeAdapter[TaskOutput | Transcript] = TypeAdapter(
    Annotated[TaskOutput | Transcript, Field(discriminator="kind")]
)


def parse_output(data: dict[str, Any]) -> TaskOutput | Transcript:
    """Rebuild a stored output (``attempt.output``) into its model."""
    return _OUTPUT.validate_python(data)
