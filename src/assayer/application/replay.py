"""Replay: re-score stored outputs with no calls to the system under test (ADR-0017).

When a number moves you need to know whether the *agent* changed or the *grading* changed.
Replay copies a run's outputs into a new run (``replay_of`` links them) and scores them with the
current checks and rubric:

* checks always recompute (they are deterministic and free);
* judgements are reused from the evidence-hash cache when the judge version is unchanged, so
  replaying with only check changes makes **zero** model calls;
* if the rubric or judge model changed, new judgements need model calls, which happen only
  with ``rejudge=True`` and are metered against the replay run's own budget.
"""

from __future__ import annotations

from dataclasses import dataclass

from assayer.application.harness import Harness
from assayer.application.run_eval import RunStore
from assayer.application.scoring import ScoreReport, ScoreRun, grader_version
from assayer.domain.errors import ConfigError
from assayer.domain.money import Money
from assayer.domain.run import RunMeta, RunStatus
from assayer.ports.system import Clock


@dataclass
class ReplayResult:
    meta: RunMeta
    copied: int
    scoring: ScoreReport


class ReplayRun:
    def __init__(self, store: RunStore, harness: Harness, *, clock: Clock) -> None:
        self._store = store
        self._harness = harness
        self._clock = clock

    async def __call__(
        self, source_run_id: str, target_run_id: str, *, rejudge: bool, budget: Money
    ) -> ReplayResult:
        source = await self._store.get_run(source_run_id)
        if source is None:
            raise ConfigError(f"unknown run {source_run_id!r}")
        if await self._store.get_run(target_run_id) is not None:
            raise ConfigError(f"run {target_run_id!r} already exists; pick a new id")
        harness = self._harness
        judge_version = harness.judge_version
        meta = source.model_copy(
            update={
                "run_id": target_run_id,
                "created_at": self._clock.now(),
                "replay_of": source_run_id,
                "judge_version": judge_version,
                "grader_version": grader_version(harness.spec, judge_version),
                "budget_micros": budget.micros,
                "models": source.models.model_copy(
                    update={
                        "judge": harness.judge_binding.settings.model
                        if harness.judge_binding
                        else None
                    }
                ),
            }
        )
        await self._store.create_or_load_run(meta, await self._store.load_cases(source_run_id))
        copied = await self._store.copy_attempts(source_run_id, target_run_id)
        scoring = await ScoreRun(self._store, harness, judge_cache_only=not rejudge)(target_run_id)
        await self._store.set_run_status(target_run_id, RunStatus.COMPLETE)
        return ReplayResult(meta=meta, copied=copied, scoring=scoring)
