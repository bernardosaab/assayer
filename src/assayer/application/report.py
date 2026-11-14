"""Build the report view model from the store. Rendering is an adapter concern."""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import Field

from assayer.application.analytics import RunAnalytics
from assayer.application.compare import Comparison
from assayer.application.queries import GradingView, ReadStore, run_analytics
from assayer.application.scoring import JUDGE_ID, judge_inputs
from assayer.domain.attempt import Attempt
from assayer.domain.base import Model
from assayer.domain.checks import CheckResult
from assayer.domain.fixtures import ConversationFixture, TaskFixture
from assayer.domain.grading import Grade
from assayer.domain.judging import JudgeVerdict
from assayer.domain.messages import ToolCall
from assayer.domain.outputs import TaskOutput, ToolEvent, Transcript, parse_output
from assayer.domain.run import RunMeta, RunStatus


class TimelineItem(Model):
    """One visible step of a conversation, for the report's timeline."""

    turn: int
    actor: str  # "user" | "agent" | "tool"
    text: str
    result: str | None = None
    error: str | None = None


class AttemptView(Model):
    attempt: Attempt
    grade: Grade
    kind: str
    tags: list[str] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    judgement: JudgeVerdict | None = None
    output_text: str = ""  # what the model actually returned
    graded_text: str = ""  # what was graded, when it differs (parsed + fields derived in code)
    turns: int | None = None
    termination: str | None = None
    tool_calls: int | None = None
    timeline: list[TimelineItem] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)


class RunReport(Model):
    meta: RunMeta
    status: RunStatus
    analytics: RunAnalytics
    attempts: list[AttemptView]
    generated_by: str


class DiffReport(Model):
    base: RunMeta
    candidate: RunMeta
    comparison: Comparison
    max_drop: float
    regressed: bool
    generated_by: str
    note: str | None = None


class ReportRenderer(Protocol):
    def render_run(self, report: RunReport) -> str: ...
    def render_diff(self, report: DiffReport) -> str: ...


def _short(value: object, limit: int = 400) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def build_timeline(transcript: Transcript) -> list[TimelineItem]:
    """Interleave user lines, agent replies, and each tool call with its result or error.

    Turn numbers match the live trace: the opening user line is turn 0, and an agent turn
    starts at the first assistant message after a user message (so interim text plus tool calls
    plus the final reply are one turn). When a tool call is matched to its event, the event's
    recorded turn is authoritative. Calls are matched to events, not looked up by id alone: some
    providers repeat or omit ids, and calls cut off by the step cap have no event.
    """
    unused = list(transcript.tool_events)
    items: list[TimelineItem] = []
    turn = 0
    after_user = True
    for message in transcript.messages:
        if message.role == "user" and message.content:
            items.append(TimelineItem(turn=turn, actor="user", text=message.content))
            after_user = True
            continue
        if message.role != "assistant":
            continue
        if after_user:
            turn += 1
            after_user = False
        for call in message.tool_calls:
            event = _take_event(unused, call, turn)
            if event is not None:
                turn = event.turn
            items.append(
                TimelineItem(
                    turn=turn,
                    actor="tool",
                    text=f"{call.name}({_short(call.arguments, 200)})",
                    result=None if event is None or event.error else _short(event.result),
                    error=event.error if event else "no result recorded (not executed)",
                )
            )
        if message.content:
            items.append(TimelineItem(turn=turn, actor="agent", text=message.content))
    return items


def _take_event(unused: list[ToolEvent], call: ToolCall, turn: int) -> ToolEvent | None:
    """Consume the best unconsumed event for ``call``.

    Preference: exact call in the same turn, exact call anywhere, same id and name in the same
    turn, same id and name anywhere. Consuming each event once keeps repeated ids from all
    resolving to one result; preferring the same turn stops a later turn's identical call from
    being taken.
    """

    def exact(e: ToolEvent) -> bool:
        return e.call == call

    def similar(e: ToolEvent) -> bool:
        return e.call.id == call.id and e.call.name == call.name

    for matches, same_turn in ((exact, True), (exact, False), (similar, True), (similar, False)):
        for index, event in enumerate(unused):
            if matches(event) and (not same_turn or event.turn == turn):
                return unused.pop(index)
    return None


def _writes(state: dict[str, object]) -> list[str]:
    writes = state.get("writes")
    return [_short(w, 300) for w in writes] if isinstance(writes, list) else []


def _kind(case: TaskFixture | ConversationFixture | None) -> str:
    if isinstance(case, ConversationFixture):
        return case.scenario.kind.value
    return "task"


async def build_run_report(
    store: ReadStore, meta: RunMeta, view_of: GradingView, *, generated_by: str
) -> RunReport:
    """Everything shown comes from one grading view: the grades, the judgement version that
    produced them, and only the check results those grades used."""
    cases = {c.id: c for c in await store.load_cases(meta.run_id)}
    views: list[AttemptView] = []
    judge_version = view_of.judge_version
    for attempt, grade in await store.list_grades(meta.run_id, view_of.grader_version):
        case = cases.get(attempt.case_id)
        checks = await store.get_check_results(attempt.attempt_id)
        if view_of.checks is not None:
            current = view_of.checks.get(attempt.case_id, set())
            checks = [c for c in checks if (c.check_id, c.version) in current]
        view = AttemptView(
            attempt=attempt,
            grade=grade,
            kind=_kind(case),
            tags=list(case.tags) if case else [],
            checks=checks,
        )
        if judge_version is not None:
            stored = await store.get_judgement(attempt.attempt_id, JUDGE_ID, judge_version)
            view = view.model_copy(update={"judgement": stored.verdict if stored else None})
        if attempt.output is not None and case is not None:
            output = parse_output(attempt.output)
            _, subject = judge_inputs(case, output)
            raw = output.raw if isinstance(output, TaskOutput) else subject
            update: dict[str, object] = {
                "output_text": raw,
                "graded_text": subject if subject != raw else "",
            }
            if isinstance(output, Transcript):
                update |= {
                    "turns": output.turns,
                    "termination": output.termination.value,
                    "tool_calls": len(output.tool_events),
                    "timeline": build_timeline(output),
                    "writes": _writes(output.final_state),
                }
            view = view.model_copy(update=update)
        views.append(view)
    views.sort(key=lambda v: (v.grade.outcome.value != "fail", v.attempt.case_id, v.attempt.repeat))
    return RunReport(
        meta=meta,
        status=await store.run_status(meta.run_id),
        analytics=await run_analytics(store, meta, view_of.grader_version),
        attempts=views,
        generated_by=generated_by,
    )
