"""Regression tests for the fourth (scoped) review of src/assayer/application."""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from assayer.adapters.store.sqlite_store import SqliteStore
from assayer.adapters.system.clock import FixedClock
from assayer.adapters.system.git import FakeVcs
from assayer.application.chat_model import RoleSettings
from assayer.application.conversation import run_conversation
from assayer.application.events import TurnEvent
from assayer.application.harness import Harness, RoleBinding
from assayer.application.provider_middleware import MeteredProvider, RetryingProvider
from assayer.application.queries import judgements_of, run_analytics
from assayer.application.replay import ReplayRun
from assayer.application.run_eval import RunEval, RunOptions
from assayer.application.scoring import ScoreRun, grader_version
from assayer.application.validate import validate_eval
from assayer.domain.attempt import AttemptStatus
from assayer.domain.errors import ErrorKind, ProviderTimeout
from assayer.domain.fixtures import CheckSpec, Criterion, TaskFixture
from assayer.domain.judging import JudgeVerdict
from assayer.domain.messages import CallContext, CompletionRequest, Message, Purpose, ToolCall
from assayer.domain.money import Money
from assayer.domain.outcome import Outcome
from assayer.domain.outputs import TaskOutput, ToolEvent
from assayer.domain.pricing import ModelPrice, PriceEntry, PriceTable
from assayer.domain.run import RunStatus
from assayer.ports.sut import AgentTurn, ChatModel, Environment, SimTurn, Tool
from tests.support.builders import a_conversation
from tests.support.fakes import StubProvider
from tests.support.harness import load_spec, scripted, task_harness


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteStore]:
    s = await SqliteStore.open(tmp_path / "r.sqlite", clock=FixedClock())
    yield s
    await s.close()


def _stricter(spec: Any) -> Any:
    return spec.model_copy(
        update={"judge": spec.judge.model_copy(update={"instructions": "be strict"})}
    )


# 1 + 2. Judge statistics and calibration use one judge version only.


async def test_judge_stats_count_only_the_current_judge_version(store: SqliteStore) -> None:
    spec = load_spec("resume_extraction")
    provider = scripted(spec)
    harness = task_harness(spec, provider, store)
    result = await RunEval(store, harness, clock=FixedClock(), vcs=FakeVcs())(
        RunOptions(run_id="r1")
    )
    new = dataclasses.replace(harness, spec=_stricter(spec))  # same run, re-judged
    await ScoreRun(store, new)("r1")
    assert len(await store.list_judgements("r1")) == 12  # both versions are stored

    analytics = await run_analytics(
        store,
        result.meta,
        grader_version(new.spec, new.judge_version),
        judge_version=new.judge_version,
    )
    assert analytics.judge.judgements == 6
    assert len(await judgements_of(store, "r1", new.judge_version)) == 6
    assert await judgements_of(store, "r1", None) == []


# 3 + 7. A replay whose judging stops on the budget is HALTED, and the un-judged attempts are
# "not measured" instead of being graded on checks alone.


async def test_replay_budget_stop_halts_and_marks_unjudged(store: SqliteStore) -> None:
    spec = load_spec("resume_extraction")
    harness = task_harness(spec, scripted(spec), store)
    await RunEval(store, harness, clock=FixedClock(), vcs=FakeVcs())(RunOptions(run_id="base"))
    new = dataclasses.replace(harness, spec=_stricter(spec))
    replayed = await ReplayRun(store, new, clock=FixedClock())(
        "base", "re", rejudge=True, budget=Money(1)
    )
    assert replayed.scoring.halted_reason is not None
    assert await store.run_status("re") is RunStatus.HALTED
    grades = await store.list_grades("re", grader_version(new.spec, new.judge_version))
    unjudged = [g for _, g in grades if g.outcome is Outcome.HARNESS_ERROR]
    assert unjudged
    assert all("judge: not run" in g.reasons[0] for g in unjudged)


# 4. "Untouched" means nothing charged: a billed timeout followed by a budget refusal is not
# released back to pending.


class _OneCall:
    async def run(self, fixture: TaskFixture, llm: ChatModel) -> TaskOutput:
        await llm.chat([Message.user("go")])
        return TaskOutput(raw="{}", parsed={})


async def test_billed_failed_call_is_not_released(store: SqliteStore) -> None:
    spec = load_spec("resume_extraction")
    price = ModelPrice(input_per_mtok=Decimal(10), output_per_mtok=Decimal(10))
    metered = MeteredProvider(
        StubProvider(raises=[ProviderTimeout("slow")]),
        store,
        PriceTable(entries=[PriceEntry(pattern="*", price=price)]),
        default_max_tokens=100,
    )

    async def no_sleep(_: float) -> None:
        return None

    provider = RetryingProvider(metered, sleep=no_sleep)
    harness = Harness(
        spec=spec,
        agent=RoleBinding(provider, RoleSettings(model="paid/m", max_tokens=100)),
        task_runner=_OneCall(),
    )
    worst = price.worst_case(
        CompletionRequest(
            model="paid/m",
            messages=(Message.user("go"),),
            context=CallContext(purpose=Purpose.AGENT),
            max_tokens=100,
        )
    )
    result = await RunEval(store, harness, clock=FixedClock(), vcs=FakeVcs())(
        RunOptions(run_id="r1", budget=Money(worst.micros * 3 // 2), concurrency=1, score=False)
    )
    attempts = {a.case_id: a for a in await store.list_attempts("r1")}
    first = attempts[spec.cases[0].id]
    assert first.status is AttemptStatus.HARNESS_ERROR  # money was spent: kept, not redone
    assert first.error_kind is ErrorKind.BUDGET_EXCEEDED
    assert result.execution.released == 0


# 5. A crashing judge is contained to its attempt.


class _CrashingJudge:
    async def judge(
        self, criteria: list[Criterion], evidence: str, subject: str, llm: ChatModel
    ) -> JudgeVerdict:
        raise KeyError("bug in a custom judge")


async def test_judge_crash_is_contained(store: SqliteStore) -> None:
    spec = load_spec("resume_extraction")
    harness = dataclasses.replace(task_harness(spec, scripted(spec), store), judge=_CrashingJudge())
    result = await RunEval(store, harness, clock=FixedClock(), vcs=FakeVcs())(
        RunOptions(run_id="r1")
    )
    assert result.status is RunStatus.COMPLETE  # scoring finished; the run did not hang OPEN
    grades = await store.list_grades("r1", grader_version(spec, harness.judge_version))
    assert len(grades) == 6
    assert all(g.outcome is Outcome.HARNESS_ERROR for _, g in grades)
    [j, *_] = await store.list_judgements("r1")
    assert j.verdict.error_kind is ErrorKind.INTERNAL


# 6. validate builds nested checks.


def test_validate_catches_typos_in_nested_checks() -> None:
    spec = load_spec("resume_extraction")
    nested = CheckSpec(use="all_of", args={"checks": [{"use": "feild_match"}]})
    report = validate_eval(spec.model_copy(update={"checks": [*spec.checks, nested]}))
    assert not report.ok
    assert any("feild_match" in i.message for i in report.issues)


# 8. Released attempts are counted as they happen.


async def test_released_counter(store: SqliteStore) -> None:
    spec = load_spec("resume_extraction")
    result = await RunEval(
        store, task_harness(spec, scripted(spec), store), clock=FixedClock(), vcs=FakeVcs()
    )(RunOptions(run_id="r1", budget=Money(1_000), concurrency=1, score=False))
    pending = [a for a in await store.list_attempts("r1") if a.status is AttemptStatus.PENDING]
    assert result.execution.released >= 1
    assert result.execution.released <= len(pending)


# 9. The live trace shows every agent message of a turn, in order.


class _Env:
    @property
    def tools(self) -> Sequence[Tool]:
        return []

    def briefing(self) -> str:
        return ""

    def snapshot(self) -> dict[str, Any]:
        return {}


class _TwoMessageAgent:
    async def respond(
        self, history: Sequence[Message], env: Environment, llm: ChatModel, *, turn: int
    ) -> AgentTurn:
        call = ToolCall(id="c1", name="lookup", arguments={})
        return AgentTurn(
            messages=(
                Message.assistant("Let me check.", (call,)),
                Message.tool("c1", "{}"),
                Message.assistant("Here is the answer."),
            ),
            tool_events=(ToolEvent(turn=turn, call=call, result={}),),
            ended_conversation=True,
        )


class _Sim:
    async def next_turn(self, *args: Any, **kwargs: Any) -> SimTurn:
        raise AssertionError("not reached")


async def test_live_trace_shows_interim_agent_text() -> None:
    events: list[TurnEvent] = []
    await run_conversation(
        a_conversation(),
        agent=_TwoMessageAgent(),
        simulator=_Sim(),
        environment=_Env(),
        agent_llm=None,  # type: ignore[arg-type]
        simulator_llm=None,  # type: ignore[arg-type]
        system_prompt="s",
        on_turn=events.append,
    )
    assert [(e.actor, e.text) for e in events if e.turn == 1] == [
        ("tool", "lookup({})"),
        ("agent", "Let me check."),
        ("agent", "Here is the answer."),
    ]
