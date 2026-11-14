"""Scoring stages: check, judge, grade (ADR-0009).

Each stage reads the store and writes only what is missing under the current version key, so
re-running a stage is free and changing one grader recomputes exactly what that grader touched:

* ``check``: key ``(attempt, check_id, check_version)``; deterministic, no model calls.
* ``judge``: key ``(attempt, judge_id, judge_version)``; model calls, metered. Identical evidence
  already judged under the same judge version is reused (the replay cache).
* ``grade``: pure and cheap, so it is always recomputed from the current check and judge rows.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Protocol

from assayer.application.events import (
    Graded,
    Halted,
    Judged,
    NullObserver,
    RunObserver,
    StageStarted,
    safe_notify,
)
from assayer.application.harness import Harness
from assayer.application.registry import build_check
from assayer.domain.attempt import Attempt, AttemptStatus
from assayer.domain.base import content_hash
from assayer.domain.checks import CheckResult
from assayer.domain.errors import BudgetExceeded, HarnessError
from assayer.domain.evalspec import EvalSpec
from assayer.domain.fixtures import ConversationFixture, TaskFixture
from assayer.domain.grading import GRADER_VERSION, Grade, grade
from assayer.domain.judging import judge_error
from assayer.domain.messages import CallContext, Purpose
from assayer.domain.outputs import TaskOutput, Transcript, parse_output
from assayer.ports.grading import Check
from assayer.ports.store import AttemptRepository, RunRepository, ScoreRepository, StoredJudgement

log = logging.getLogger(__name__)

JUDGE_ID = "llm_judge"
Case = TaskFixture | ConversationFixture


class ScoringStore(AttemptRepository, RunRepository, ScoreRepository, Protocol):
    pass


@dataclass
class ScoreReport:
    checks_computed: int = 0
    judgements_computed: int = 0
    judgements_reused: int = 0
    judge_errors: int = 0
    judgements_skipped: int = 0
    grades: int = 0
    halted_reason: str | None = None
    notes: list[str] = field(default_factory=list)


def checks_for(spec: EvalSpec, case: Case) -> list[Check]:
    return [build_check(s) for s in spec.checks_for(case)]


def grader_version(spec: EvalSpec, judge_version: str | None) -> str:
    """Changes whenever the grading inputs change: grade logic, any check version, the judge."""
    check_versions = sorted(
        {(c.check_id, c.version) for case in spec.cases for c in checks_for(spec, case)}
    )
    return content_hash(GRADER_VERSION, check_versions, judge_version)


def judge_inputs(case: Case, output: TaskOutput | Transcript) -> tuple[str, str]:
    """(evidence, subject): what quotes are verified against, and what is being graded.

    Extraction: quotes may come from the source document or the output, so both are evidence.
    The subject is the *parsed* output, which includes fields derived in code, so the judge and
    the checks grade the same thing; unparseable output falls back to the raw text.
    Conversation: the rendered transcript is both evidence and subject.
    """
    if isinstance(output, Transcript):
        text = output.render()
        return text, text
    source = case.input_doc if isinstance(case, TaskFixture) else ""
    subject = (
        json.dumps(output.parsed, ensure_ascii=False) if output.parsed is not None else output.raw
    )
    return f"SOURCE DOCUMENT:\n{source}\n\nEXTRACTED OUTPUT:\n{subject}", subject


class ScoreRun:
    def __init__(
        self,
        store: ScoringStore,
        harness: Harness,
        *,
        judge_cache_only: bool = False,
        observer: RunObserver | None = None,
    ) -> None:
        self._store = store
        self._harness = harness
        self._cache_only = judge_cache_only
        self._observer: RunObserver = observer or NullObserver()

    async def __call__(self, run_id: str, *, judge: bool = True) -> ScoreReport:
        spec = self._harness.spec
        cases = {c.id: c for c in await self._store.load_cases(run_id)}
        attempts = await self._store.list_attempts(run_id)
        report = ScoreReport()
        succeeded = [a for a in attempts if a.status is AttemptStatus.SUCCEEDED]

        for attempt in succeeded:
            report.checks_computed += await self._check(attempt, cases[attempt.case_id])
        if judge and self._harness.judge is not None and self._harness.judge_binding is not None:
            await self._judge_all(run_id, succeeded, cases, report)
        elif spec.judge is not None and judge:
            report.notes.append("eval defines a judge rubric but no judge model is configured")

        if report.judgements_skipped:
            report.notes.append(
                f"{report.judgements_skipped} judgement(s) not in cache were skipped; "
                "grades for those attempts ignore the judge (use --rejudge to call it)"
            )
        version = grader_version(spec, self._harness.judge_version)
        for attempt in attempts:
            if attempt.status in (AttemptStatus.SUCCEEDED, AttemptStatus.HARNESS_ERROR):
                graded = await self._grade(attempt, cases[attempt.case_id])
                await self._store.save_grade(attempt.attempt_id, version, graded)
                report.grades += 1
                safe_notify(
                    self._observer,
                    Graded(
                        attempt.case_id,
                        attempt.repeat,
                        graded.outcome.value,
                        tuple(graded.reasons),
                    ),
                )
        return report

    async def _check(self, attempt: Attempt, case: Case) -> int:
        assert attempt.output is not None
        output = parse_output(attempt.output)
        existing = {
            (r.check_id, r.version) for r in await self._store.get_check_results(attempt.attempt_id)
        }
        computed = 0
        for check in checks_for(self._harness.spec, case):
            if (check.check_id, check.version) in existing:
                continue
            try:
                result = check.evaluate(case, output)
            except Exception as exc:  # a broken check must not abort scoring of the whole run
                log.exception("check %s crashed on attempt %s", check.check_id, attempt.attempt_id)
                result = CheckResult.failed(
                    check.check_id, check.version, f"check crashed: {type(exc).__name__}: {exc}"
                )
            await self._store.save_check_result(attempt.attempt_id, result)
            computed += 1
        return computed

    async def _judge_all(
        self, run_id: str, attempts: list[Attempt], cases: dict[str, Case], report: ScoreReport
    ) -> None:
        h = self._harness
        assert h.judge is not None
        assert h.judge_binding is not None
        version = h.judge_version
        assert version is not None
        # Decide the work first, so the progress total counts only what will be judged.
        todo: list[Attempt] = []
        for attempt in attempts:
            if not h.spec.criteria_for(cases[attempt.case_id]):
                continue
            existing = await self._store.get_judgement(attempt.attempt_id, JUDGE_ID, version)
            if existing is None or existing.verdict.verdict_final is None:
                todo.append(attempt)
        safe_notify(self._observer, StageStarted("judge", len(todo)))
        for attempt in todo:
            case = cases[attempt.case_id]
            criteria = h.spec.criteria_for(case)
            assert attempt.output is not None
            evidence, subject = judge_inputs(case, parse_output(attempt.output))
            evidence_hash = content_hash(evidence, subject, criteria)

            cached = await self._store.find_judgement_by_evidence(JUDGE_ID, version, evidence_hash)
            if cached is not None:
                verdict = cached.verdict
                report.judgements_reused += 1
            elif self._cache_only:
                report.judgements_skipped += 1
                safe_notify(
                    self._observer,
                    Judged(attempt.case_id, attempt.repeat, None, 0, 0, cached=False, skipped=True),
                )
                continue
            else:
                llm = h.judge_binding.bind(
                    CallContext(
                        purpose=Purpose.JUDGE,
                        run_id=run_id,
                        case_id=attempt.case_id,
                        attempt_id=attempt.attempt_id,
                        judgement_key=f"{JUDGE_ID}:{version}",
                    )
                )
                try:
                    verdict = await h.judge.judge(criteria, evidence, subject, llm)
                except BudgetExceeded as exc:
                    report.halted_reason = f"budget exhausted while judging ({exc})"
                    safe_notify(self._observer, Halted(report.halted_reason))
                    return
                except HarnessError as exc:
                    verdict = judge_error(exc.kind, str(exc))
                report.judgements_computed += 1
            if verdict.verdict_final is None:
                report.judge_errors += 1
            safe_notify(
                self._observer,
                Judged(
                    attempt.case_id,
                    attempt.repeat,
                    verdict.verdict_final,
                    verdict.reasks,
                    len(verdict.dropped),
                    cached=cached is not None,
                ),
            )
            await self._store.save_judgement(
                StoredJudgement(
                    attempt_id=attempt.attempt_id,
                    judge_id=JUDGE_ID,
                    judge_version=version,
                    evidence_hash=evidence_hash,
                    judge_model=h.judge_binding.settings.model,
                    verdict=verdict,
                )
            )

    async def _grade(self, attempt: Attempt, case: Case) -> Grade:
        if attempt.status is AttemptStatus.HARNESS_ERROR:
            return grade(attempt_status=attempt.status, agent_error=None, checks=[], judge=None)
        assert attempt.output is not None
        output = parse_output(attempt.output)
        current = {(c.check_id, c.version) for c in checks_for(self._harness.spec, case)}
        results = [
            r
            for r in await self._store.get_check_results(attempt.attempt_id)
            if (r.check_id, r.version) in current
        ]
        judgement = None
        version = self._harness.judge_version
        if version is not None and self._harness.spec.criteria_for(case):
            stored = await self._store.get_judgement(attempt.attempt_id, JUDGE_ID, version)
            judgement = stored.verdict if stored else None
        return grade(
            attempt_status=attempt.status,
            agent_error=output.agent_error,
            checks=results,
            judge=judgement,
        )
