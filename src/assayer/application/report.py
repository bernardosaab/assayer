"""Build the report view model from the store. Rendering is an adapter concern."""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import Field

from assayer.application.analytics import RunAnalytics
from assayer.application.compare import Comparison
from assayer.application.queries import ReadStore, run_analytics
from assayer.application.scoring import JUDGE_ID, judge_inputs
from assayer.domain.attempt import Attempt
from assayer.domain.base import Model
from assayer.domain.checks import CheckResult
from assayer.domain.fixtures import ConversationFixture, TaskFixture
from assayer.domain.grading import Grade
from assayer.domain.judging import JudgeVerdict
from assayer.domain.outputs import Transcript, parse_output
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
    output_text: str = ""
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


class ReportRenderer(Protocol):
    def render_run(self, report: RunReport) -> str: ...
    def render_diff(self, report: DiffReport) -> str: ...


def _short(value: object, limit: int = 400) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def build_timeline(transcript: Transcript) -> list[TimelineItem]:
    """Interleave user lines, agent replies, and each tool call with its result or error."""
    events = {e.call.id: e for e in transcript.tool_events}
    items: list[TimelineItem] = []
    turn = 0
    for message in transcript.messages:
        if message.role == "user" and message.content:
            items.append(TimelineItem(turn=turn, actor="user", text=message.content))
        elif message.role == "assistant":
            for call in message.tool_calls:
                event = events.get(call.id)
                turn = event.turn if event else turn
                items.append(
                    TimelineItem(
                        turn=turn,
                        actor="tool",
                        text=f"{call.name}({_short(call.arguments, 200)})",
                        result=None if event is None or event.error else _short(event.result),
                        error=event.error if event else None,
                    )
                )
            if message.content:
                items.append(TimelineItem(turn=turn, actor="agent", text=message.content))
                turn += 1
    return items


def _writes(state: dict[str, object]) -> list[str]:
    writes = state.get("writes")
    return [_short(w, 300) for w in writes] if isinstance(writes, list) else []


def _kind(case: TaskFixture | ConversationFixture | None) -> str:
    if isinstance(case, ConversationFixture):
        return case.scenario.kind.value
    return "task"


async def build_run_report(
    store: ReadStore, meta: RunMeta, grader_version: str, *, generated_by: str
) -> RunReport:
    cases = {c.id: c for c in await store.load_cases(meta.run_id)}
    views: list[AttemptView] = []
    judge_version = meta.judge_version
    for attempt, grade in await store.list_grades(meta.run_id, grader_version):
        case = cases.get(attempt.case_id)
        view = AttemptView(
            attempt=attempt,
            grade=grade,
            kind=_kind(case),
            tags=list(case.tags) if case else [],
            checks=await store.get_check_results(attempt.attempt_id),
        )
        if judge_version is not None:
            stored = await store.get_judgement(attempt.attempt_id, JUDGE_ID, judge_version)
            view = view.model_copy(update={"judgement": stored.verdict if stored else None})
        if attempt.output is not None and case is not None:
            output = parse_output(attempt.output)
            _, subject = judge_inputs(case, output)
            update: dict[str, object] = {"output_text": subject}
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
        analytics=await run_analytics(store, meta, grader_version),
        attempts=views,
        generated_by=generated_by,
    )
