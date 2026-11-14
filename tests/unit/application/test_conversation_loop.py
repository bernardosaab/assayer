"""Stop conditions of the conversation loop live in code; each one is exercised here."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from assayer.application.conversation import run_conversation
from assayer.domain.fixtures import Persona, Scenario
from assayer.domain.messages import CompletionResponse, Message
from assayer.domain.outputs import TerminationReason, Transcript
from assayer.ports.sut import AgentTurn, ChatModel, Environment, SimTurn, Tool
from tests.support.builders import a_conversation


class _Env:
    @property
    def tools(self) -> Sequence[Tool]:
        return []

    def snapshot(self) -> dict[str, Any]:
        return {"final": {"ok": True}}


class _NoLlm:
    model = "none"
    calls_made = 0

    async def chat(self, messages: Sequence[Message], **_: Any) -> CompletionResponse:
        raise AssertionError("not used")


class _Agent:
    def __init__(self, *, ended: bool = False, error: str | None = None) -> None:
        self.ended, self.error = ended, error

    async def respond(
        self, history: Sequence[Message], env: Environment, llm: ChatModel, *, turn: int
    ) -> AgentTurn:
        return AgentTurn(
            messages=(Message.assistant(f"reply {turn}"),),
            ended_conversation=self.ended,
            error=self.error,
        )


class _Sim:
    def __init__(self, turns: list[SimTurn]) -> None:
        self.turns = turns

    async def next_turn(
        self,
        persona: Persona,
        scenario: Scenario,
        history: Sequence[Message],
        llm: ChatModel,
        *,
        turn: int,
    ) -> SimTurn:
        return self.turns.pop(0)


async def _run(agent: _Agent, sim: _Sim, max_turns: int = 3) -> Transcript:
    fixture = a_conversation()
    fixture = fixture.model_copy(
        update={"scenario": fixture.scenario.model_copy(update={"max_turns": max_turns})}
    )
    return await run_conversation(
        fixture,
        agent=agent,
        simulator=sim,
        environment=_Env(),
        agent_llm=_NoLlm(),
        simulator_llm=_NoLlm(),
        system_prompt="sys",
    )


@pytest.mark.parametrize(
    ("agent", "sim_turns", "expected", "turns"),
    [
        (_Agent(), [SimTurn(done=True, reason="goal_met")], TerminationReason.GOAL_MET, 1),
        (_Agent(), [SimTurn(done=True, reason="gave_up")], TerminationReason.USER_GAVE_UP, 1),
        (_Agent(ended=True), [], TerminationReason.AGENT_ENDED, 1),
        (_Agent(error="loop"), [], TerminationReason.AGENT_TOOL_LOOP, 1),
        (_Agent(), [SimTurn(message="Hi!")], TerminationReason.SIMULATOR_LOOP, 1),
        (
            _Agent(),
            [SimTurn(message="a"), SimTurn(message="b")],
            TerminationReason.TURN_LIMIT,
            3,
        ),
    ],
)
async def test_termination(
    agent: _Agent, sim_turns: list[SimTurn], expected: TerminationReason, turns: int
) -> None:
    transcript = await _run(agent, _Sim(sim_turns))
    assert transcript.termination is expected
    assert transcript.turns == turns
    assert transcript.final_state == {"final": {"ok": True}}
    if expected is TerminationReason.AGENT_TOOL_LOOP:
        assert transcript.agent_error == "loop"
