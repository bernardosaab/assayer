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
) -> Transcript:
    scenario = fixture.scenario
    history: list[Message] = [Message.system(system_prompt), Message.user(scenario.opening_message)]
    events: list[ToolEvent] = []
    last_user = scenario.opening_message
    termination = TerminationReason.TURN_LIMIT
    agent_error: str | None = None
    turn = 0

    for turn in range(1, scenario.max_turns + 1):
        step = await agent.respond(history, environment, agent_llm, turn=turn)
        history.extend(step.messages)
        events.extend(step.tool_events)
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

    return Transcript(
        messages=tuple(history),
        tool_events=tuple(events),
        final_state=environment.snapshot(),
        termination=termination,
        turns=turn,
        agent_error=agent_error,
    )
