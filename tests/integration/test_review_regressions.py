"""Regression tests for findings from the code review of v0.2.0.

Each test reproduces the failure scenario the review described; all failed before the fix.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from assayer.adapters.providers.openai_compat import OpenAICompatProvider
from assayer.adapters.store.sqlite_store import SqliteStore
from assayer.adapters.system.clock import FixedClock
from assayer.adapters.system.git import FakeVcs
from assayer.application.conversation import run_conversation
from assayer.application.events import AttemptFinished, Judged, RecordingObserver, StageStarted
from assayer.application.queries import grading_view
from assayer.application.registry import build_task_runner
from assayer.application.report import build_timeline
from assayer.application.run_eval import RunEval, RunOptions
from assayer.application.scoring import judge_inputs
from assayer.application.validate import validate_eval
from assayer.domain.base import Model, content_hash
from assayer.domain.errors import ConfigError
from assayer.domain.fixtures import Criterion
from assayer.domain.messages import (
    CallContext,
    CompletionRequest,
    Message,
    Purpose,
    ToolCall,
)
from assayer.domain.money import Money
from assayer.domain.outputs import TaskOutput, TerminationReason, ToolEvent, Transcript
from assayer.ports.sut import AgentTurn, ChatModel, Environment, SimTurn, Tool
from tests.support.builders import a_conversation, a_task
from tests.support.harness import load_spec, scripted, task_harness


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteStore]:
    s = await SqliteStore.open(tmp_path / "r.sqlite", clock=FixedClock())
    yield s
    await s.close()


# 1. An environment written before briefing() existed must still work.


class _LegacyEnv:
    @property
    def tools(self) -> Sequence[Tool]:
        return []

    def snapshot(self) -> dict[str, Any]:
        return {}


class _OneReplyAgent:
    async def respond(
        self, history: Sequence[Message], env: Environment, llm: ChatModel, *, turn: int
    ) -> AgentTurn:
        return AgentTurn(messages=(Message.assistant("hi"),), ended_conversation=True)


class _NoSim:
    async def next_turn(self, *args: Any, **kwargs: Any) -> SimTurn:
        raise AssertionError("not reached")


async def test_environment_without_briefing_still_runs() -> None:
    transcript = await run_conversation(
        a_conversation(),
        agent=_OneReplyAgent(),
        simulator=_NoSim(),
        environment=_LegacyEnv(),  # type: ignore[arg-type]
        agent_llm=None,  # type: ignore[arg-type]
        simulator_llm=None,  # type: ignore[arg-type]
        system_prompt="sys",
    )
    assert transcript.termination is TerminationReason.AGENT_ENDED


# 2 + 10. Timeline: pairing in order (ids may repeat), capped calls, consistent turn numbers.


def test_timeline_pairs_repeated_ids_in_order_and_numbers_turns() -> None:
    first = ToolCall(id="call_0", name="get_pto_balance", arguments={"employee_id": "E1"})
    second = ToolCall(id="call_0", name="submit_pto_request", arguments={"days": 2})
    capped = ToolCall(id="call_1", name="escalate_to_hr", arguments={})
    transcript = Transcript(
        messages=(
            Message.system("s"),
            Message.user("hello"),
            Message.assistant("Let me check.", (first,)),
            Message.tool("call_0", '{"days": 12}'),
            Message.assistant("You have 12 days."),
            Message.user("book it"),
            Message.assistant(None, (second, capped)),
        ),
        tool_events=(
            ToolEvent(turn=1, call=first, result={"days": 12}),
            ToolEvent(turn=2, call=second, error="timeout"),
        ),
        termination=TerminationReason.AGENT_TOOL_LOOP,
    )
    items = build_timeline(transcript)
    assert [(i.turn, i.actor) for i in items] == [
        (0, "user"),
        (1, "tool"),
        (1, "agent"),  # interim text, tool call and reply are ONE turn
        (1, "agent"),
        (1, "user"),
        (2, "tool"),
        (2, "tool"),
    ]
    tools = [i for i in items if i.actor == "tool"]
    assert tools[0].result is not None
    assert "12" in tools[0].result
    assert tools[1].error == "timeout"  # not the first call's result, despite the same id
    assert tools[2].error is not None
    assert "not executed" in tools[2].error


async def test_provider_fallback_tool_call_ids_are_unique() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        call = {"type": "function", "function": {"name": "t", "arguments": "{}"}}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": None, "tool_calls": [call, call]}}
                ]
            },
        )

    provider = OpenAICompatProvider("ollama", "http://x/v1", transport=httpx.MockTransport(handler))
    request = CompletionRequest(
        model="ollama/m",
        messages=(Message.user("hi"),),
        context=CallContext(purpose=Purpose.AGENT),
    )
    ids = [c.id for _ in range(2) for c in (await provider.complete(request)).message.tool_calls]
    await provider.aclose()
    assert len(set(ids)) == 4


# 3. The judge grades what the checks grade: parsed output including derived fields.


def test_judge_subject_includes_derived_fields() -> None:
    output = TaskOutput(
        raw='{"first_job_start_year": 2016}',
        parsed={"first_job_start_year": 2016, "years_experience": 10},
    )
    evidence, subject = judge_inputs(a_task(), output)
    assert '"years_experience": 10' in subject
    assert '"years_experience": 10' in evidence
    unparsed = TaskOutput(raw="prose", agent_error="not JSON")
    assert judge_inputs(a_task(), unparsed)[1] == "prose"


# 4 + 5. Grading views are consistent; diff warns when versions differ (see CLI test below).


async def test_grading_view_falls_back_and_says_so(store: SqliteStore) -> None:
    spec = load_spec("resume_extraction")
    harness = task_harness(spec, scripted(spec), store)
    result = await RunEval(store, harness, clock=FixedClock(), vcs=FakeVcs())(
        RunOptions(run_id="r1")
    )
    view = await grading_view(store, result.meta, spec)
    assert view.current
    assert view.judge_version == spec.judge_version("fake/judge")
    assert view.checks is not None
    assert len(view.checks["jane-doe"]) == 4

    assert spec.judge is not None
    stricter = spec.model_copy(
        update={"judge": spec.judge.model_copy(update={"instructions": "be strict"})}
    )
    fallback = await grading_view(store, result.meta, stricter)
    assert not fallback.current  # new graders never applied to r1: shown as originally graded
    assert fallback.grader_version == result.meta.grader_version


# 6. Adding a defaulted field to a model must not change existing version hashes.


def test_content_hash_includes_defaults_so_changed_defaults_change_versions() -> None:
    # Second review: hashing without defaults hid default changes and broke resuming old runs.
    class V1(Model):
        steps: int = 5

    class V2(Model):
        steps: int = 8

    assert content_hash(V1()) != content_hash(V2())
    # Models and their dumps hash the same, so call sites may pass either.
    assert content_hash([Criterion(id="a", text="t")]) == content_hash(
        [Criterion(id="a", text="t").model_dump()]
    )


# 7. A bad derive rule is a clean config error, caught by validate before any spend.


@pytest.mark.parametrize(
    ("derive", "message"),
    [
        ({"years": {"years_since": "first_job_start_year"}}, "invalid derive rule"),
        ({"years": {"years_since": "typo", "as_of": 2026}}, "does not ask for"),
    ],
)
def test_bad_derive_rules_are_config_errors(derive: dict[str, Any], message: str) -> None:
    schema = {"type": "object", "properties": {"first_job_start_year": {"type": "integer"}}}
    with pytest.raises(ConfigError, match=message):
        build_task_runner("extraction", system_prompt="x", output_schema=schema, derive=derive)


def test_validate_reports_bad_derive() -> None:
    spec = load_spec("resume_extraction")
    assert spec.task is not None
    broken = spec.model_copy(
        update={
            "task": spec.task.model_copy(
                update={
                    "args": {"derive": {"years_experience": {"years_since": "nope", "as_of": 1}}}
                }
            )
        }
    )
    report = validate_eval(broken)
    assert not report.ok
    assert any("does not ask for" in i.message for i in report.issues)


# 8. Attempts released untouched by a budget halt are not reported as executed errors.


async def test_budget_released_attempts_are_not_errors(store: SqliteStore) -> None:
    spec = load_spec("resume_extraction")
    observer = RecordingObserver()
    result = await RunEval(
        store,
        task_harness(spec, scripted(spec), store),
        clock=FixedClock(),
        vcs=FakeVcs(),
        observer=observer,
    )(RunOptions(run_id="r1", budget=Money(1_000), concurrency=1, score=False))
    finished = observer.of(AttemptFinished)
    released = [e for e in finished if e.released]
    assert released  # the attempt refused by the budget went back to pending
    assert result.execution.executed == len(finished) - len(released)
    assert result.execution.harness_errors == 0


# 9. The judge progress total counts only the work that will run.


async def test_judge_progress_total_matches_work_on_resume(store: SqliteStore) -> None:
    spec = load_spec("resume_extraction")
    harness = task_harness(spec, scripted(spec), store)
    first = RecordingObserver()
    await RunEval(store, harness, clock=FixedClock(), vcs=FakeVcs(), observer=first)(
        RunOptions(run_id="r1")
    )
    [stage] = [e for e in first.of(StageStarted) if e.stage == "judge"]
    assert stage.total == len(first.of(Judged)) == 6

    again = RecordingObserver()
    await RunEval(store, harness, clock=FixedClock(), vcs=FakeVcs(), observer=again)(
        RunOptions(run_id="r1")
    )
    [stage] = [e for e in again.of(StageStarted) if e.stage == "judge"]
    assert stage.total == 0  # everything already judged: no bar that never completes
    assert again.of(Judged) == []
