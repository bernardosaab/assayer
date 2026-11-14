"""A reference tool-calling agent: the system under test for conversation evals.

Per user turn it loops ``model -> tool calls -> tool results -> model`` until the model answers
in plain text, with a hard cap on tool calls per turn that lives in code, not in the prompt.

Tool failures (unknown tool, invalid arguments, a tool error such as a timeout) are returned to
the model as an ``{"error": ...}`` result and recorded on the :class:`ToolEvent`. How the agent
*reports* such failures to the user is exactly what the chaos scenarios grade.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from assayer.application.registry import AGENTS
from assayer.domain.messages import Message, ToolCall
from assayer.domain.outputs import ToolEvent
from assayer.ports.sut import AgentTurn, ChatModel, Environment, Tool, ToolError


def _invoke(tools: dict[str, Tool], call: ToolCall) -> tuple[Any, str | None]:
    tool = tools.get(call.name)
    if tool is None:
        return None, f"unknown tool {call.name!r}"
    if set(call.arguments) == {"_raw"}:
        return None, f"arguments are not a JSON object: {call.arguments['_raw']!r}"
    try:
        return tool.invoke(call.arguments), None
    except ToolError as exc:
        return None, str(exc)


class ToolAgent:
    def __init__(self, *, max_tool_steps: int = 5) -> None:
        self._max_steps = max_tool_steps

    async def respond(
        self, history: Sequence[Message], env: Environment, llm: ChatModel, *, turn: int
    ) -> AgentTurn:
        tools = {t.spec.name: t for t in env.tools}
        specs = [t.spec for t in env.tools]
        produced: list[Message] = []
        events: list[ToolEvent] = []
        calls_this_turn = 0

        while True:
            reply = await llm.chat([*history, *produced], tools=specs, turn=turn)
            message = reply.message
            produced.append(message)
            if not message.tool_calls:
                return AgentTurn(messages=tuple(produced), tool_events=tuple(events))
            calls_this_turn += len(message.tool_calls)
            if calls_this_turn > self._max_steps:
                return AgentTurn(
                    messages=tuple(produced),
                    tool_events=tuple(events),
                    error=f"exceeded {self._max_steps} tool calls in one turn",
                )
            for call in message.tool_calls:
                result, error = _invoke(tools, call)
                events.append(ToolEvent(turn=turn, call=call, result=result, error=error))
                payload = {"error": error} if error else result
                produced.append(Message.tool(call.id, json.dumps(payload, default=str)))


@AGENTS.register("tool_agent")
def _tool_agent(*, max_tool_steps: int = 5, **_: Any) -> ToolAgent:
    return ToolAgent(max_tool_steps=max_tool_steps)
