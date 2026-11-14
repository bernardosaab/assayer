"""End-to-end task pipeline against the real store and the scripted provider."""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from assayer.adapters.providers.faults import FaultInjectingProvider
from assayer.adapters.store.sqlite_store import SqliteStore
from assayer.adapters.system.clock import FixedClock
from assayer.adapters.system.git import FakeVcs
from assayer.application.queries import run_analytics
from assayer.application.run_eval import RunEval, RunOptions
from assayer.application.scoring import JUDGE_ID, grader_version
from assayer.domain.errors import ProviderUnavailable
from assayer.domain.fixtures import CheckSpec
from assayer.domain.money import Money
from assayer.domain.outcome import Outcome
from assayer.domain.run import RunStatus
from tests.support.harness import load_spec, scripted, task_harness

EXPECTED = {
    "jane-doe": Outcome.PASS,
    "carlos-ruiz": Outcome.FAIL,  # previous employer reported as current (trap)
    "mei-chen": Outcome.PASS,
    "amara-okafor": Outcome.PASS,  # F1 0.91 >= 0.8; judge's fabricated quote dropped
    "tomasz-nowak": Outcome.PASS,
    "priya-sharma": Outcome.FAIL,  # prose instead of JSON
}


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteStore]:
    s = await SqliteStore.open(tmp_path / "r.sqlite", clock=FixedClock())
    yield s
    await s.close()


async def _outcomes(store: SqliteStore, run_id: str, version: str) -> dict[str, Outcome]:
    return {a.case_id: g.outcome for a, g in await store.list_grades(run_id, version)}


async def test_full_run_grades_every_case_as_scripted(store: SqliteStore) -> None:
    spec = load_spec("resume_extraction")
    harness = task_harness(spec, scripted(spec), store)
    result = await RunEval(store, harness, clock=FixedClock(), vcs=FakeVcs())(
        RunOptions(run_id="r1", concurrency=3)
    )
    assert result.status is RunStatus.COMPLETE
    version = grader_version(spec, harness.judge_version)
    assert await _outcomes(store, "r1", version) == EXPECTED

    judgements = {
        j.attempt_id: j.verdict for j in await store.list_judgements("r1") if j.judge_id == JUDGE_ID
    }
    by_case = {a.case_id: a.attempt_id for a in await store.list_attempts("r1")}
    assert judgements[by_case["mei-chen"]].reasks == 1  # contradiction re-asked
    assert len(judgements[by_case["amara-okafor"]].dropped) == 1  # fabricated quote dropped
    assert judgements[by_case["carlos-ruiz"]].verdict_final == "fail"

    analytics = await run_analytics(store, result.meta, version)
    assert analytics.rates.pass_rate == pytest.approx(4 / 6)
    assert analytics.total_cost > Money.zero()
    assert analytics.cost_per_success == Money(analytics.total_cost.micros // 4)
    assert ("check:trap_absent", 1) in analytics.failure_causes


async def test_rerun_is_free_and_identical(store: SqliteStore) -> None:
    spec = load_spec("resume_extraction")
    provider = scripted(spec)
    harness = task_harness(spec, provider, store)
    run = RunEval(store, harness, clock=FixedClock(), vcs=FakeVcs())
    await run(RunOptions(run_id="r1"))
    calls, ledger_rows = len(provider.calls), len(await store.ledger("r1"))

    again = await run(RunOptions(run_id="r1"))
    assert again.resumed
    assert again.execution.executed == 0
    assert len(provider.calls) == calls
    assert len(await store.ledger("r1")) == ledger_rows


async def test_changing_one_check_recomputes_only_that_check(store: SqliteStore) -> None:
    spec = load_spec("resume_extraction")
    provider = scripted(spec)
    run = RunEval(store, task_harness(spec, provider, store), clock=FixedClock(), vcs=FakeVcs())
    await run(RunOptions(run_id="r1"))
    calls = len(provider.calls)

    stricter = [
        CheckSpec(use="set_f1", args={"field": "skills", "threshold": 0.95})
        if c.use == "set_f1"
        else c
        for c in spec.checks
    ]
    new_spec = spec.model_copy(update={"checks": stricter})
    harness = dataclasses.replace(task_harness(new_spec, provider, store), spec=new_spec)
    result = await RunEval(store, harness, clock=FixedClock(), vcs=FakeVcs())(
        RunOptions(run_id="r1")
    )
    assert result.scoring is not None
    assert result.scoring.checks_computed == 6  # one new set_f1 result per attempt
    assert result.scoring.judgements_computed == 0  # judge untouched
    assert len(provider.calls) == calls  # no model calls at all
    outcomes = await _outcomes(store, "r1", grader_version(new_spec, harness.judge_version))
    assert outcomes["amara-okafor"] is Outcome.FAIL  # 0.91 < 0.95 now


async def test_budget_halt_then_resume_with_more_budget(store: SqliteStore) -> None:
    spec = load_spec("resume_extraction")
    provider = scripted(spec)
    run = RunEval(store, task_harness(spec, provider, store), clock=FixedClock(), vcs=FakeVcs())

    halted = await run(RunOptions(run_id="r1", budget=Money(1_000), concurrency=1))
    assert halted.status is RunStatus.HALTED
    assert halted.execution.halted_reason is not None
    assert "budget" in halted.execution.halted_reason
    assert await store.committed("r1") <= Money(1_000)
    pending = [a for a in await store.list_attempts("r1") if a.status.value == "pending"]
    assert pending  # untouched work waits, it is not marked failed

    done = await run(RunOptions(run_id="r1", budget=Money.from_usd("1"), concurrency=1))
    assert done.status is RunStatus.COMPLETE
    harness = task_harness(spec, provider, store)
    version = grader_version(spec, harness.judge_version)
    assert await _outcomes(store, "r1", version) == EXPECTED


async def test_circuit_breaker_stops_a_dead_endpoint(store: SqliteStore) -> None:
    spec = load_spec("resume_extraction")
    dead = FaultInjectingProvider(
        scripted(spec), fail_from_call=(1, lambda: ProviderUnavailable("connection refused"))
    )
    harness = task_harness(spec, dead, store)
    result = await RunEval(store, harness, clock=FixedClock(), vcs=FakeVcs())(
        RunOptions(run_id="r1", concurrency=1, circuit_breaker=3)
    )
    assert result.status is RunStatus.HALTED
    assert result.execution.harness_errors == 3
    assert result.execution.halted_reason is not None
    assert "circuit breaker" in result.execution.halted_reason
    # Refused connections were never billed: every reservation was released.
    assert await store.committed("r1") == Money.zero()
    analytics = await run_analytics(store, result.meta, grader_version(spec, harness.judge_version))
    assert analytics.rates.graded == 0  # harness errors never count toward the rate
    assert analytics.rates.harness_errors == 3


async def test_retry_errors_recovers_after_outage(store: SqliteStore) -> None:
    spec = load_spec("resume_extraction")
    flaky = FaultInjectingProvider(scripted(spec), errors={1: lambda: ProviderUnavailable("blip")})
    harness = task_harness(spec, flaky, store)
    run = RunEval(store, harness, clock=FixedClock(), vcs=FakeVcs())
    first = await run(RunOptions(run_id="r1", concurrency=1))
    assert first.execution.harness_errors == 1
    second = await run(RunOptions(run_id="r1", concurrency=1, retry_errors=True))
    assert second.retried == 1
    assert second.status is RunStatus.COMPLETE
    version = grader_version(spec, harness.judge_version)
    assert await _outcomes(store, "r1", version) == EXPECTED


async def test_repeats_give_pass_hat_k(store: SqliteStore) -> None:
    spec = load_spec("posting_extraction")
    harness = task_harness(spec, scripted(spec), store)
    result = await RunEval(store, harness, clock=FixedClock(), vcs=FakeVcs())(
        RunOptions(run_id="r1", repeats=3)
    )
    analytics = await run_analytics(store, result.meta, grader_version(spec, harness.judge_version))
    assert analytics.rates.graded == 9
    assert analytics.pass_hat_k == pytest.approx(2 / 3)  # data-analyst always hits the trap
    assert analytics.pass_at_k == pytest.approx(2 / 3)
