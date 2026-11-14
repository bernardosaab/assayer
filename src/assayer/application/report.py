"""Build the report view model from the store. Rendering is an adapter concern."""

from __future__ import annotations

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
