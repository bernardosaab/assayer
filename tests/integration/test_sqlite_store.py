"""Behavioural tests of the SQLite store against a real database file."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from contextlib import closing
from pathlib import Path

import pytest

from assayer.adapters.store.migrator import current_version, migrate, split_statements
from assayer.adapters.store.sqlite_store import SqliteStore
from assayer.adapters.system.clock import FixedClock
from assayer.domain.attempt import AttemptStatus
from assayer.domain.checks import CheckResult
from assayer.domain.errors import BudgetExceeded, ConfigError, ErrorKind, IllegalTransition
from assayer.domain.grading import Grade
from assayer.domain.judging import JudgeResponse, judge_error, settle_verdict
from assayer.domain.messages import Purpose, Usage
from assayer.domain.money import Money
from assayer.domain.outcome import Outcome
from assayer.domain.run import RunStatus
from assayer.ports.store import StoredJudgement
from tests.support.builders import a_conversation, a_run, a_task


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteStore]:
    s = await SqliteStore.open(tmp_path / "r.sqlite", clock=FixedClock(), session_id="s1")
    yield s
    await s.close()


async def _seed(store: SqliteStore, repeats: int = 2, budget: str = "1.00") -> None:
    await store.create_or_load_run(
        a_run(repeats=repeats, budget_micros=Money.from_usd(budget).micros),
        [a_task("c1"), a_conversation("c2")],
    )
    await store.ensure_attempts("run-1", repeats)


# --------------------------------------------------------------------------- migrations


def test_migrations_are_idempotent(tmp_path: Path) -> None:
    with closing(sqlite3.connect(tmp_path / "m.sqlite", isolation_level=None)) as conn:
        assert migrate(conn) == 1
        assert migrate(conn) == 1
        assert current_version(conn) == 1


def test_split_statements_rejects_trailing_garbage() -> None:
    assert split_statements("-- c\nCREATE TABLE t (x);\n") == ["-- c\nCREATE TABLE t (x);"]
    with pytest.raises(ValueError, match="incomplete"):
        split_statements("CREATE TABLE t (x)")


# --------------------------------------------------------------------------- runs


async def test_create_then_load_run_round_trips_cases(store: SqliteStore) -> None:
    await _seed(store)
    cases = await store.load_cases("run-1")
    assert [c.id for c in cases] == ["c1", "c2"]
    assert cases[1].kind == "conversation"
    again = await store.create_or_load_run(a_run(repeats=2), [])
    assert again.run_id == "run-1"
    assert await store.run_status("run-1") is RunStatus.OPEN
    await store.set_run_status("run-1", RunStatus.COMPLETE)
    [(meta, status)] = await store.list_runs()
    assert (meta.run_id, status) == ("run-1", RunStatus.COMPLETE)


async def test_same_run_id_with_different_config_is_refused(store: SqliteStore) -> None:
    await _seed(store)
    with pytest.raises(ConfigError, match="different configuration"):
        await store.create_or_load_run(a_run(config_hash="other"), [])


async def test_unknown_run(store: SqliteStore) -> None:
    assert await store.get_run("nope") is None
    with pytest.raises(ConfigError):
        await store.run_status("nope")


# --------------------------------------------------------------------------- attempts


async def test_ensure_attempts_is_idempotent(store: SqliteStore) -> None:
    await _seed(store, repeats=2)
    await store.ensure_attempts("run-1", 2)
    attempts = await store.list_attempts("run-1")
    assert len(attempts) == 4
    assert {a.status for a in attempts} == {AttemptStatus.PENDING}


async def test_claim_is_exclusive_and_finish_is_terminal(store: SqliteStore) -> None:
    await _seed(store)
    attempt = await store.claim_attempt("run-1", "c1", 0)
    assert attempt is not None
    assert attempt.status is AttemptStatus.IN_PROGRESS
    assert attempt.try_count == 1
    assert await store.claim_attempt("run-1", "c1", 0) is None

    await store.finish_attempt(attempt.attempt_id, {"raw": "x"})
    assert await store.claim_attempt("run-1", "c1", 0) is None
    with pytest.raises(IllegalTransition):
        await store.finish_attempt(attempt.attempt_id, {"raw": "y"})
    [done] = [a for a in await store.list_attempts("run-1") if a.attempt_id == attempt.attempt_id]
    assert done.output == {"raw": "x"}


async def test_release_returns_attempt_to_pending_without_counting_a_try(
    store: SqliteStore,
) -> None:
    await _seed(store)
    attempt = await store.claim_attempt("run-1", "c1", 0)
    assert attempt is not None
    await store.release_attempt(attempt.attempt_id)
    again = await store.claim_attempt("run-1", "c1", 0)
    assert again is not None
    assert again.try_count == 1


async def test_recovery_resets_uncharged_and_surfaces_charged(tmp_path: Path) -> None:
    path = tmp_path / "crash.sqlite"
    first = await SqliteStore.open(path, clock=FixedClock(), session_id="dead-process")
    await _seed(first, repeats=1)
    uncharged = await first.claim_attempt("run-1", "c1", 0)
    charged = await first.claim_attempt("run-1", "c2", 0)
    assert uncharged is not None
    assert charged is not None
    await first.reserve(
        run_id="run-1",
        purpose=Purpose.AGENT,
        model="m",
        amount=Money(500),
        attempt_id=charged.attempt_id,
    )
    await first.close()  # simulate the process dying mid-call

    second = await SqliteStore.open(path, clock=FixedClock(), session_id="new-process")
    report = await second.recover_interrupted("run-1")
    assert (report.reset, report.interrupted_after_charge, report.orphaned_reservations) == (
        1,
        1,
        1,
    )
    by_case = {a.case_id: a for a in await second.list_attempts("run-1")}
    assert by_case["c1"].status is AttemptStatus.PENDING
    assert by_case["c2"].status is AttemptStatus.HARNESS_ERROR
    assert by_case["c2"].error_kind is ErrorKind.INTERRUPTED_AFTER_CHARGE
    # Orphaned money still counts against the budget.
    assert await second.committed("run-1") == Money(500)
    await second.close()


async def test_retry_errors_respects_max_tries_and_clears_scores(store: SqliteStore) -> None:
    await _seed(store, repeats=1)
    attempt = await store.claim_attempt("run-1", "c1", 0)
    assert attempt is not None
    await store.fail_attempt(attempt.attempt_id, ErrorKind.PROVIDER_TIMEOUT, "slow")
    await store.save_grade(attempt.attempt_id, "g", Grade(outcome=Outcome.HARNESS_ERROR))
    assert await store.retry_errors("run-1", max_tries=1) == 0
    assert await store.retry_errors("run-1", max_tries=3) == 1
    assert await store.get_grade(attempt.attempt_id, "g") is None
    [a, _] = await store.list_attempts("run-1")
    assert a.status is AttemptStatus.PENDING
    assert a.error_kind is None


# --------------------------------------------------------------------------- ledger


async def test_reserve_settle_release_accounting(store: SqliteStore) -> None:
    await _seed(store, budget="0.001")  # 1000 micros
    a = await store.reserve(run_id="run-1", purpose=Purpose.AGENT, model="m", amount=Money(600))
    with pytest.raises(BudgetExceeded):
        await store.reserve(run_id="run-1", purpose=Purpose.JUDGE, model="m", amount=Money(500))
    await store.settle(a, Money(100), Usage(prompt_tokens=10, completion_tokens=5))
    await store.settle(a, Money(999), None)  # idempotent: second settle is a no-op
    assert await store.committed("run-1") == Money(100)
    b = await store.reserve(run_id="run-1", purpose=Purpose.JUDGE, model="m", amount=Money(900))
    await store.release(b)
    assert await store.committed("run-1") == Money(100)
    entries = await store.ledger("run-1")
    assert [e.status for e in entries] == ["settled", "released"]
    assert entries[0].prompt_tokens == 10


async def test_raise_budget(store: SqliteStore) -> None:
    await _seed(store, budget="0")
    with pytest.raises(BudgetExceeded):
        await store.reserve(run_id="run-1", purpose=Purpose.AGENT, model="m", amount=Money(1))
    await store.raise_budget("run-1", Money(10))
    await store.reserve(run_id="run-1", purpose=Purpose.AGENT, model="m", amount=Money(1))
    meta = await store.get_run("run-1")
    assert meta is not None
    assert meta.budget_micros == 10


async def test_reserve_on_unknown_run(store: SqliteStore) -> None:
    with pytest.raises(ConfigError):
        await store.reserve(run_id="x", purpose=Purpose.AGENT, model="m", amount=Money(1))


# --------------------------------------------------------------------------- scores


async def test_check_results_are_write_once(store: SqliteStore) -> None:
    await _seed(store)
    attempt = await store.claim_attempt("run-1", "c1", 0)
    assert attempt is not None
    await store.save_check_result(attempt.attempt_id, CheckResult.ok("c", "1", n=1))
    await store.save_check_result(attempt.attempt_id, CheckResult.failed("c", "1", "changed"))
    [result] = await store.get_check_results(attempt.attempt_id)
    assert result.passed
    assert result.detail == {"n": 1}


async def test_judgement_cache_and_harness_error_overwrite(store: SqliteStore) -> None:
    await _seed(store)
    attempt = await store.claim_attempt("run-1", "c1", 0)
    assert attempt is not None
    errored = StoredJudgement(
        attempt_id=attempt.attempt_id,
        judge_id="j",
        judge_version="v",
        evidence_hash="e",
        judge_model="m",
        verdict=judge_error(ErrorKind.JUDGE_INCONSISTENT, "x"),
    )
    await store.save_judgement(errored)
    assert await store.find_judgement_by_evidence("j", "v", "e") is None

    done = errored.model_copy(
        update={"verdict": settle_verdict(JudgeResponse(reasoning="ok", verdict="pass"), "")}
    )
    await store.save_judgement(done)
    await store.save_judgement(errored)  # cannot overwrite a finished judgement
    stored = await store.get_judgement(attempt.attempt_id, "j", "v")
    assert stored is not None
    assert stored.verdict.verdict_final == "pass"
    assert await store.find_judgement_by_evidence("j", "v", "e") == stored
    assert await store.list_judgements("run-1") == [stored]


async def test_grades_upsert_and_list(store: SqliteStore) -> None:
    await _seed(store, repeats=1)
    attempt = await store.claim_attempt("run-1", "c1", 0)
    assert attempt is not None
    await store.save_grade(attempt.attempt_id, "g", Grade(outcome=Outcome.FAIL, reasons=["x"]))
    await store.save_grade(attempt.attempt_id, "g", Grade(outcome=Outcome.PASS))
    [(listed, grade)] = await store.list_grades("run-1", "g")
    assert listed.case_id == "c1"
    assert grade.outcome is Outcome.PASS
    assert await store.list_grades("run-1", "other") == []
