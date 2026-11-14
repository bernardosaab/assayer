"""The conversation loop: agent and simulated user take turns until a stop condition.

Stop conditions live here, in code, not in either model's judgement alone:

* the simulator reports ``done`` (goal met or gave up);
* the agent ends the conversation itself;
* the scenario's ``max_turns`` is reached;
* the agent exceeds its per-turn tool-call budget (graded as a failure);
* the simulator repeats itself verbatim, a cheap loop detector.

The simulator sees the persona and scenario, never the expected outcomes, so it cannot steer
the agent toward the answer key.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from assayer.application.events import TurnEvent
from assayer.domain.fixtures import ConversationFixture
from assayer.domain.messages import Message
from assayer.domain.outputs import TerminationReason, ToolEvent, Transcript
from assayer.ports.sut import ChatModel, ConversationalAgent, Environment, UserSimulator


async def run_conversation(
    fixture: ConversationFixture,
    *,
    agent: ConversationalAgent,
    simulator: UserSimulator,
    environment: Environment,
    agent_llm: ChatModel,
    simulator_llm: ChatModel,
    system_prompt: str,
    on_turn: Callable[[TurnEvent], None] | None = None,
) -> Transcript:
    scenario = fixture.scenario
    # Environments written before briefing() existed (or returning None) still work.
    brief = getattr(environment, "briefing", None)
    briefing = ((brief() if callable(brief) else None) or "").strip()
    system = "\n\n".join(p for p in (system_prompt.rstrip(), briefing) if p)
    history: list[Message] = [Message.system(system), Message.user(scenario.opening_message)]
    events: list[ToolEvent] = []
    last_user = scenario.opening_message
    termination = TerminationReason.TURN_LIMIT
    agent_error: str | None = None
    turn = 0
    emit = on_turn or (lambda _event: None)
    emit(TurnEvent(fixture.id, 0, "user", scenario.opening_message))

    for turn in range(1, scenario.max_turns + 1):
        step = await agent.respond(history, environment, agent_llm, turn=turn)
        history.extend(step.messages)
        events.extend(step.tool_events)
        # Every message of the turn, in order (interim text, tool calls, final reply), so the
        # live trace shows the same sequence as the report's timeline.
        pending_events = list(step.tool_events)
        for message in step.messages:
            if message.role != "assistant":
                continue
            for call in message.tool_calls:
                tool_event = next((e for e in pending_events if e.call == call), None)
                if tool_event is not None:
                    pending_events.remove(tool_event)
                emit(
                    TurnEvent(
                        fixture.id,
                        turn,
                        "tool",
                        f"{call.name}({json.dumps(call.arguments, default=str)})",
                        error=tool_event.error if tool_event else "not executed",
                    )
                )
            if message.content:
                emit(TurnEvent(fixture.id, turn, "agent", message.content))
        if step.error is not None:
            termination, agent_error = TerminationReason.AGENT_TOOL_LOOP, step.error
            break
        if step.ended_conversation:
            termination = TerminationReason.AGENT_ENDED
            break
        if turn == scenario.max_turns:
            break

        sim = await simulator.next_turn(
            fixture.persona, scenario, history[1:], simulator_llm, turn=turn
        )
        if sim.done:
            termination = (
                TerminationReason.GOAL_MET
                if sim.reason == "goal_met"
                else TerminationReason.USER_GAVE_UP
            )
            break
        if sim.message.strip() == last_user.strip():
            termination = TerminationReason.SIMULATOR_LOOP
            break
        last_user = sim.message
        history.append(Message.user(sim.message))
        emit(TurnEvent(fixture.id, turn, "user", sim.message))

    return Transcript(
        messages=tuple(history),
        tool_events=tuple(events),
        final_state=environment.snapshot(),
        termination=termination,
        turns=turn,
        agent_error=agent_error,
    )
