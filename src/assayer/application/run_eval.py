"""The ``run`` use case: create or resume a run, execute it, score it.

A run is identified by its id. Re-running the same id resumes it: finished attempts are skipped,
interrupted ones are recovered, and only missing work is done. Reusing an id with a different
configuration (prompt, models, fixtures, repeats) is refused, so results never mix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from assayer.application.events import RunObserver
from assayer.application.execute import ExecutionReport, Executor
from assayer.application.harness import Harness
from assayer.application.scoring import ScoreReport, ScoreRun, ScoringStore, grader_version
from assayer.domain.base import content_hash
from assayer.domain.money import Money
from assayer.domain.run import ModelRoles, RunMeta, RunStatus
from assayer.ports.store import CostLedger, RecoveryReport
from assayer.ports.system import Clock, VcsInfo


class RunStore(ScoringStore, CostLedger, Protocol):
    pass


@dataclass(frozen=True)
class RunOptions:
    run_id: str
    repeats: int = 1
    budget: Money = field(default_factory=lambda: Money.from_usd("1.00"))
    concurrency: int = 4
    circuit_breaker: int = 5
    retry_errors: bool = False
    max_tries: int = 3
    score: bool = True
    judge: bool = True
    round: int = 1


@dataclass
class RunResult:
    meta: RunMeta
    status: RunStatus
    recovery: RecoveryReport
    execution: ExecutionReport
    scoring: ScoreReport | None
    retried: int
    resumed: bool


def build_run_meta(harness: Harness, options: RunOptions, clock: Clock, vcs: VcsInfo) -> RunMeta:
    spec = harness.spec
    models = ModelRoles(
        agent=harness.agent.settings.model,
        simulator=harness.simulator.settings.model if harness.simulator else None,
        judge=harness.judge_binding.settings.model if harness.judge_binding else None,
    )
    judge_version = harness.judge_version
    config_hash = content_hash(
        spec.name,
        spec.prompt_version,
        models,
        options.repeats,
        options.round,
        sorted(case.fixture_hash for case in spec.cases),
    )
    return RunMeta(
        run_id=options.run_id,
        eval_name=spec.name,
        created_at=clock.now(),
        git_sha=vcs.sha(),
        git_dirty=vcs.dirty(),
        prompt_version=spec.prompt_version,
        judge_version=judge_version,
        grader_version=grader_version(spec, judge_version),
        models=models,
        repeats=options.repeats,
        round=options.round,
        budget_micros=options.budget.micros,
        config_hash=config_hash,
    )


class RunEval:
    def __init__(
        self,
        store: RunStore,
        harness: Harness,
        *,
        clock: Clock,
        vcs: VcsInfo,
        observer: RunObserver | None = None,
    ) -> None:
        self._observer = observer
        self._store = store
        self._harness = harness
        self._clock = clock
        self._vcs = vcs

    async def __call__(self, options: RunOptions) -> RunResult:
        wanted = build_run_meta(self._harness, options, self._clock, self._vcs)
        resumed = await self._store.get_run(options.run_id) is not None
        meta = await self._store.create_or_load_run(wanted, self._harness.spec.cases)
        if options.budget.micros > meta.budget_micros:
            await self._store.raise_budget(meta.run_id, options.budget)

        await self._store.ensure_attempts(meta.run_id, meta.repeats)
        recovery = await self._store.recover_interrupted(meta.run_id)
        retried = (
            await self._store.retry_errors(meta.run_id, max_tries=options.max_tries)
            if options.retry_errors
            else 0
        )
        await self._store.set_run_status(meta.run_id, RunStatus.OPEN)

        execution = await Executor(
            self._store,
            self._harness,
            concurrency=options.concurrency,
            circuit_breaker=options.circuit_breaker,
            observer=self._observer,
        ).execute(meta.run_id)

        scoring = None
        if options.score:
            scoring = await ScoreRun(self._store, self._harness, observer=self._observer)(
                meta.run_id, judge=options.judge
            )

        halted = execution.halted_reason or (scoring.halted_reason if scoring else None)
        unfinished = any(
            a.status.value in ("pending", "in_progress")
            for a in await self._store.list_attempts(meta.run_id)
        )
        status = RunStatus.HALTED if halted or unfinished else RunStatus.COMPLETE
        await self._store.set_run_status(meta.run_id, status)
        return RunResult(
            meta=await self._store.get_run(meta.run_id) or meta,
            status=status,
            recovery=recovery,
            execution=execution,
            scoring=scoring,
            retried=retried,
            resumed=resumed,
        )
