"""Read-side queries shared by the CLI and the report renderer."""

from __future__ import annotations

from typing import Protocol

from assayer.application.analytics import RunAnalytics, analyze
from assayer.domain.attempt import Attempt
from assayer.domain.grading import Grade
from assayer.domain.outcome import Outcome
from assayer.domain.run import RunMeta
from assayer.ports.store import AttemptRepository, CostLedger, RunRepository, ScoreRepository


class ReadStore(RunRepository, AttemptRepository, ScoreRepository, CostLedger, Protocol):
    pass


async def run_analytics(store: ReadStore, meta: RunMeta, grader_version: str) -> RunAnalytics:
    return analyze(
        await store.list_grades(meta.run_id, grader_version),
        cases=await store.load_cases(meta.run_id),
        ledger=await store.ledger(meta.run_id),
        judgements=await store.list_judgements(meta.run_id),
    )


async def failures(
    store: ReadStore, run_id: str, grader_version: str
) -> list[tuple[Attempt, Grade]]:
    return [
        (a, g)
        for a, g in await store.list_grades(run_id, grader_version)
        if g.outcome is not Outcome.PASS
    ]
