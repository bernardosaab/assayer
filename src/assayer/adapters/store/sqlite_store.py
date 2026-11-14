"""SQLite implementation of every store port (ADR-0003).

Concurrency model: **one connection, one thread.** Every operation is a small synchronous
function submitted to a single-worker executor, so writes from many asyncio tasks are
serialized without locks inside the process. Across processes, ``BEGIN IMMEDIATE`` on the
read-modify-write transactions (claim, reserve) plus ``busy_timeout`` keep the invariants.
A lock that outlives the timeout surfaces as :class:`StoreLocked`, a harness error.
"""

from __future__ import annotations

import asyncio
import functools
import json
import sqlite3
import uuid
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import TypeAdapter

from assayer.adapters.store.migrator import migrate
from assayer.domain.attempt import Attempt, AttemptStatus, recover
from assayer.domain.base import canonical_json
from assayer.domain.checks import CheckResult
from assayer.domain.errors import (
    BudgetExceeded,
    ConfigError,
    ErrorKind,
    IllegalTransition,
    StoreLocked,
)
from assayer.domain.fixtures import ConversationFixture, Fixture, TaskFixture
from assayer.domain.grading import Grade
from assayer.domain.judging import JudgeVerdict
from assayer.domain.messages import Purpose, Usage
from assayer.domain.money import Money
from assayer.domain.run import RunMeta, RunStatus
from assayer.ports.store import CalibrationLabel, LedgerEntry, RecoveryReport, StoredJudgement
from assayer.ports.system import Clock

T = TypeVar("T")

_FIXTURE: TypeAdapter[TaskFixture | ConversationFixture] = TypeAdapter(Fixture)

# Committed spend of a ledger row: settled -> actual; reserved/orphaned -> reserved; released -> 0.
_EFFECTIVE_COST = (
    "CASE status WHEN 'settled' THEN actual_micros WHEN 'released' THEN 0 ELSE reserved_micros END"
)


class SqliteStore:
    """Implements ``ResultStore`` (runs, attempts, cost ledger, scores)."""

    def __init__(self, path: Path | str, *, clock: Clock, session_id: str | None = None) -> None:
        self._path = str(path)
        self._clock = clock
        self.session_id = session_id or uuid.uuid4().hex
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="assayer-sqlite")
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    async def open(
        cls, path: Path | str, *, clock: Clock, session_id: str | None = None
    ) -> SqliteStore:
        store = cls(path, clock=clock, session_id=session_id)
        await store._call(store._connect)
        return store

    def _connect(self) -> None:
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            self._path, isolation_level=None, check_same_thread=False, timeout=5.0
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        migrate(conn)
        self._conn = conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._call(self._conn.close)
            self._conn = None
        self._executor.shutdown(wait=True)

    async def _call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                self._executor, functools.partial(fn, *args, **kwargs)
            )
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if "locked" in message or "busy" in message:
                raise StoreLocked(str(exc)) from exc
            raise

    @property
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("store is not open")
        return self._conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        conn = self._db
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    def _now(self) -> str:
        return self._clock.now().isoformat()

    # ------------------------------------------------------------------ runs

    async def create_or_load_run(
        self, meta: RunMeta, cases: Sequence[TaskFixture | ConversationFixture]
    ) -> RunMeta:
        return await self._call(self._create_or_load_run, meta, list(cases))

    def _create_or_load_run(
        self, meta: RunMeta, cases: list[TaskFixture | ConversationFixture]
    ) -> RunMeta:
        with self._tx() as db:
            row = db.execute(
                "SELECT config_hash, meta_json FROM run WHERE run_id = ?", (meta.run_id,)
            ).fetchone()
            if row is not None:
                if row["config_hash"] != meta.config_hash:
                    raise ConfigError(
                        f"run {meta.run_id!r} exists with a different configuration "
                        f"({row['config_hash']} != {meta.config_hash}); use a new run id "
                        "or --force-new-round"
                    )
                return RunMeta.model_validate_json(row["meta_json"])
            db.execute(
                "INSERT INTO run (run_id, eval_name, created_at, config_hash, budget_micros, "
                "status, replay_of, meta_json) VALUES (?, ?, ?, ?, ?, 'open', ?, ?)",
                (
                    meta.run_id,
                    meta.eval_name,
                    meta.created_at.isoformat(),
                    meta.config_hash,
                    meta.budget_micros,
                    meta.replay_of,
                    meta.model_dump_json(),
                ),
            )
            db.executemany(
                "INSERT INTO run_case (run_id, case_id, ordinal, fixture_hash, fixture_json) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (meta.run_id, case.id, i, case.fixture_hash, canonical_json(case))
                    for i, case in enumerate(cases)
                ],
            )
            return meta

    async def get_run(self, run_id: str) -> RunMeta | None:
        return await self._call(self._get_run, run_id)

    def _get_run(self, run_id: str) -> RunMeta | None:
        row = self._db.execute("SELECT meta_json FROM run WHERE run_id = ?", (run_id,)).fetchone()
        return None if row is None else RunMeta.model_validate_json(row["meta_json"])

    async def list_runs(self) -> list[tuple[RunMeta, RunStatus]]:
        return await self._call(self._list_runs)

    def _list_runs(self) -> list[tuple[RunMeta, RunStatus]]:
        rows = self._db.execute(
            "SELECT meta_json, status FROM run ORDER BY created_at DESC, run_id"
        ).fetchall()
        return [(RunMeta.model_validate_json(r["meta_json"]), RunStatus(r["status"])) for r in rows]

    async def run_status(self, run_id: str) -> RunStatus:
        return await self._call(self._run_status, run_id)

    def _run_status(self, run_id: str) -> RunStatus:
        row = self._db.execute("SELECT status FROM run WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise ConfigError(f"unknown run {run_id!r}")
        return RunStatus(row["status"])

    async def set_run_status(self, run_id: str, status: RunStatus) -> None:
        await self._call(
            self._db.execute, "UPDATE run SET status = ? WHERE run_id = ?", (status.value, run_id)
        )

    async def load_cases(self, run_id: str) -> list[TaskFixture | ConversationFixture]:
        return await self._call(self._load_cases, run_id)

    def _load_cases(self, run_id: str) -> list[TaskFixture | ConversationFixture]:
        rows = self._db.execute(
            "SELECT fixture_json FROM run_case WHERE run_id = ? ORDER BY ordinal", (run_id,)
        ).fetchall()
        return [_FIXTURE.validate_json(r["fixture_json"]) for r in rows]

    # ------------------------------------------------------------------ attempts

    async def ensure_attempts(self, run_id: str, repeats: int) -> None:
        await self._call(self._ensure_attempts, run_id, repeats)

    def _ensure_attempts(self, run_id: str, repeats: int) -> None:
        with self._tx() as db:
            case_ids = [
                r["case_id"]
                for r in db.execute(
                    "SELECT case_id FROM run_case WHERE run_id = ? ORDER BY ordinal", (run_id,)
                )
            ]
            db.executemany(
                "INSERT INTO attempt (run_id, case_id, repeat, status) VALUES (?, ?, ?, 'pending') "
                "ON CONFLICT (run_id, case_id, repeat) DO NOTHING",
                [(run_id, c, r) for c in case_ids for r in range(repeats)],
            )

    async def recover_interrupted(self, run_id: str) -> RecoveryReport:
        return await self._call(self._recover_interrupted, run_id)

    def _recover_interrupted(self, run_id: str) -> RecoveryReport:
        with self._tx() as db:
            orphaned = db.execute(
                "UPDATE cost_ledger SET status = 'orphaned' "
                "WHERE run_id = ? AND status = 'reserved' AND session_id != ?",
                (run_id, self.session_id),
            ).rowcount
            reset = interrupted = 0
            rows = db.execute(
                f"SELECT a.attempt_id, (SELECT COALESCE(SUM({_EFFECTIVE_COST}), 0) "  # noqa: S608
                "FROM cost_ledger l WHERE l.attempt_id = a.attempt_id) AS cost "
                "FROM attempt a WHERE a.run_id = ? AND a.status = 'in_progress'",
                (run_id,),
            ).fetchall()
            for row in rows:
                status, kind = recover(AttemptStatus.IN_PROGRESS, charged=row["cost"] > 0)
                if kind is ErrorKind.INTERRUPTED_AFTER_CHARGE:
                    interrupted += 1
                    message = "interrupted after spending; rerun with --retry-errors"
                else:
                    reset += 1
                    message = None
                db.execute(
                    "UPDATE attempt SET status = ?, error_kind = ?, error_msg = ? "
                    "WHERE attempt_id = ?",
                    (status.value, kind.value if kind else None, message, row["attempt_id"]),
                )
            return RecoveryReport(
                reset=reset, interrupted_after_charge=interrupted, orphaned_reservations=orphaned
            )

    async def claim_attempt(self, run_id: str, case_id: str, repeat: int) -> Attempt | None:
        return await self._call(self._claim_attempt, run_id, case_id, repeat)

    def _claim_attempt(self, run_id: str, case_id: str, repeat: int) -> Attempt | None:
        with self._tx() as db:
            claimed = db.execute(
                "UPDATE attempt SET status = 'in_progress', try_count = try_count + 1, "
                "started_at = ?, finished_at = NULL "
                "WHERE run_id = ? AND case_id = ? AND repeat = ? AND status = 'pending' "
                "RETURNING attempt_id",
                (self._now(), run_id, case_id, repeat),
            ).fetchone()
            if claimed is None:
                return None
            return self._attempt(db, claimed["attempt_id"])

    def _transition(self, attempt_id: int, sql: str, params: tuple[Any, ...]) -> None:
        with self._tx() as db:
            if db.execute(sql, params).rowcount != 1:
                status = db.execute(
                    "SELECT status FROM attempt WHERE attempt_id = ?", (attempt_id,)
                ).fetchone()
                raise IllegalTransition(
                    f"attempt {attempt_id} is {status['status'] if status else 'missing'}"
                )

    async def finish_attempt(self, attempt_id: int, output: dict[str, Any]) -> None:
        await self._call(
            self._transition,
            attempt_id,
            "UPDATE attempt SET status = 'succeeded', output_json = ?, finished_at = ?, "
            "error_kind = NULL, error_msg = NULL WHERE attempt_id = ? AND status = 'in_progress'",
            (canonical_json(output), self._now(), attempt_id),
        )

    async def fail_attempt(self, attempt_id: int, kind: ErrorKind, message: str) -> None:
        await self._call(
            self._transition,
            attempt_id,
            "UPDATE attempt SET status = 'harness_error', error_kind = ?, error_msg = ?, "
            "finished_at = ? WHERE attempt_id = ? AND status = 'in_progress'",
            (kind.value, message[:2000], self._now(), attempt_id),
        )

    async def release_attempt(self, attempt_id: int) -> None:
        await self._call(
            self._transition,
            attempt_id,
            "UPDATE attempt SET status = 'pending', try_count = MAX(try_count - 1, 0), "
            "started_at = NULL WHERE attempt_id = ? AND status = 'in_progress'",
            (attempt_id,),
        )

    async def retry_errors(self, run_id: str, *, max_tries: int) -> int:
        return await self._call(self._retry_errors, run_id, max_tries)

    def _retry_errors(self, run_id: str, max_tries: int) -> int:
        with self._tx() as db:
            ids = [
                r["attempt_id"]
                for r in db.execute(
                    "SELECT attempt_id FROM attempt "
                    "WHERE run_id = ? AND status = 'harness_error' AND try_count < ?",
                    (run_id, max_tries),
                )
            ]
            for attempt_id in ids:
                # Scores computed from the failed execution are stale once it is redone.
                for table in ("grade", "check_result"):
                    db.execute(f"DELETE FROM {table} WHERE attempt_id = ?", (attempt_id,))  # noqa: S608
                for table in ("calibration", "judgement"):
                    db.execute(f"DELETE FROM {table} WHERE attempt_id = ?", (attempt_id,))  # noqa: S608
                db.execute(
                    "UPDATE attempt SET status = 'pending', error_kind = NULL, error_msg = NULL "
                    "WHERE attempt_id = ?",
                    (attempt_id,),
                )
            return len(ids)

    async def copy_attempts(self, source_run_id: str, target_run_id: str) -> int:
        return await self._call(self._copy_attempts, source_run_id, target_run_id)

    def _copy_attempts(self, source_run_id: str, target_run_id: str) -> int:
        """Copy finished attempts (outputs, timings, errors) into a replay run."""
        with self._tx() as db:
            return db.execute(
                "INSERT INTO attempt (run_id, case_id, repeat, status, output_json, error_kind, "
                "error_msg, try_count, started_at, finished_at) "
                "SELECT ?, case_id, repeat, status, output_json, error_kind, error_msg, try_count, "
                "started_at, finished_at FROM attempt "
                "WHERE run_id = ? AND status IN ('succeeded', 'harness_error') "
                "ON CONFLICT (run_id, case_id, repeat) DO NOTHING",
                (target_run_id, source_run_id),
            ).rowcount

    async def list_attempts(self, run_id: str) -> list[Attempt]:
        return await self._call(self._list_attempts, run_id)

    _ATTEMPT_SELECT = (
        f"SELECT a.*, (SELECT COALESCE(SUM({_EFFECTIVE_COST}), 0) FROM cost_ledger l "  # noqa: S608
        "WHERE l.attempt_id = a.attempt_id) AS cost_micros FROM attempt a"
    )

    def _list_attempts(self, run_id: str) -> list[Attempt]:
        rows = self._db.execute(
            f"{self._ATTEMPT_SELECT} WHERE a.run_id = ? ORDER BY a.attempt_id", (run_id,)
        ).fetchall()
        return [_attempt_from_row(r) for r in rows]

    def _attempt(self, db: sqlite3.Connection, attempt_id: int) -> Attempt:
        row = db.execute(f"{self._ATTEMPT_SELECT} WHERE a.attempt_id = ?", (attempt_id,)).fetchone()
        return _attempt_from_row(row)

    # ------------------------------------------------------------------ cost ledger

    async def reserve(
        self,
        *,
        run_id: str,
        purpose: Purpose,
        model: str,
        amount: Money,
        attempt_id: int | None = None,
        judgement_key: str | None = None,
    ) -> int:
        return await self._call(
            self._reserve,
            run_id=run_id,
            purpose=purpose,
            model=model,
            amount=amount,
            attempt_id=attempt_id,
            judgement_key=judgement_key,
        )

    def _committed(self, db: sqlite3.Connection, run_id: str) -> Money:
        row = db.execute(
            f"SELECT COALESCE(SUM({_EFFECTIVE_COST}), 0) FROM cost_ledger WHERE run_id = ?",  # noqa: S608
            (run_id,),
        ).fetchone()
        return Money(int(row[0]))

    def _reserve(
        self,
        *,
        run_id: str,
        purpose: Purpose,
        model: str,
        amount: Money,
        attempt_id: int | None,
        judgement_key: str | None,
    ) -> int:
        with self._tx() as db:
            run = db.execute("SELECT budget_micros FROM run WHERE run_id = ?", (run_id,)).fetchone()
            if run is None:
                raise ConfigError(f"unknown run {run_id!r}")
            budget, committed = Money(run["budget_micros"]), self._committed(db, run_id)
            if committed + amount > budget:
                raise BudgetExceeded(committed=committed, requested=amount, budget=budget)
            cursor = db.execute(
                "INSERT INTO cost_ledger (run_id, attempt_id, judgement_key, purpose, model, "
                "status, reserved_micros, session_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'reserved', ?, ?, ?)",
                (
                    run_id,
                    attempt_id,
                    judgement_key,
                    purpose.value,
                    model,
                    amount.micros,
                    self.session_id,
                    self._now(),
                ),
            )
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    async def settle(self, entry_id: int, actual: Money, usage: Usage | None) -> None:
        await self._call(
            self._db.execute,
            "UPDATE cost_ledger SET status = 'settled', actual_micros = ?, prompt_tokens = ?, "
            "completion_tokens = ?, settled_at = ? "
            "WHERE entry_id = ? AND status IN ('reserved', 'orphaned')",
            (
                actual.micros,
                usage.prompt_tokens if usage else None,
                usage.completion_tokens if usage else None,
                self._now(),
                entry_id,
            ),
        )

    async def release(self, entry_id: int) -> None:
        await self._call(
            self._db.execute,
            "UPDATE cost_ledger SET status = 'released', actual_micros = 0, settled_at = ? "
            "WHERE entry_id = ? AND status IN ('reserved', 'orphaned')",
            (self._now(), entry_id),
        )

    async def committed(self, run_id: str) -> Money:
        return await self._call(self._committed, self._db, run_id)

    async def ledger(self, run_id: str) -> list[LedgerEntry]:
        return await self._call(self._ledger, run_id)

    def _ledger(self, run_id: str) -> list[LedgerEntry]:
        rows = self._db.execute(
            "SELECT * FROM cost_ledger WHERE run_id = ? ORDER BY entry_id", (run_id,)
        ).fetchall()
        return [
            LedgerEntry(
                entry_id=r["entry_id"],
                run_id=r["run_id"],
                attempt_id=r["attempt_id"],
                judgement_key=r["judgement_key"],
                purpose=Purpose(r["purpose"]),
                model=r["model"],
                status=r["status"],
                reserved_micros=r["reserved_micros"],
                actual_micros=r["actual_micros"],
                prompt_tokens=r["prompt_tokens"],
                completion_tokens=r["completion_tokens"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    async def raise_budget(self, run_id: str, budget: Money) -> None:
        await self._call(self._raise_budget, run_id, budget)

    def _raise_budget(self, run_id: str, budget: Money) -> None:
        with self._tx() as db:
            row = db.execute("SELECT meta_json FROM run WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise ConfigError(f"unknown run {run_id!r}")
            meta = RunMeta.model_validate_json(row["meta_json"])
            updated = meta.model_copy(update={"budget_micros": budget.micros})
            db.execute(
                "UPDATE run SET budget_micros = ?, meta_json = ? WHERE run_id = ?",
                (budget.micros, updated.model_dump_json(), run_id),
            )

    # ------------------------------------------------------------------ scores

    async def get_check_results(self, attempt_id: int) -> list[CheckResult]:
        return await self._call(self._get_check_results, attempt_id)

    def _get_check_results(self, attempt_id: int) -> list[CheckResult]:
        rows = self._db.execute(
            "SELECT * FROM check_result WHERE attempt_id = ? ORDER BY check_id", (attempt_id,)
        ).fetchall()
        return [
            CheckResult(
                check_id=r["check_id"],
                version=r["check_version"],
                passed=bool(r["passed"]),
                detail=json.loads(r["detail_json"]),
            )
            for r in rows
        ]

    async def save_check_result(self, attempt_id: int, result: CheckResult) -> None:
        await self._call(
            self._db.execute,
            "INSERT INTO check_result (attempt_id, check_id, check_version, passed, detail_json) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            (
                attempt_id,
                result.check_id,
                result.version,
                int(result.passed),
                canonical_json(result.detail),
            ),
        )

    async def get_judgement(
        self, attempt_id: int, judge_id: str, judge_version: str
    ) -> StoredJudgement | None:
        return await self._call(self._get_judgement, attempt_id, judge_id, judge_version)

    def _get_judgement(
        self, attempt_id: int, judge_id: str, judge_version: str
    ) -> StoredJudgement | None:
        row = self._db.execute(
            "SELECT * FROM judgement WHERE attempt_id = ? AND judge_id = ? AND judge_version = ?",
            (attempt_id, judge_id, judge_version),
        ).fetchone()
        return None if row is None else _judgement_from_row(row)

    async def find_judgement_by_evidence(
        self, judge_id: str, judge_version: str, evidence_hash: str
    ) -> StoredJudgement | None:
        return await self._call(self._find_judgement, judge_id, judge_version, evidence_hash)

    def _find_judgement(
        self, judge_id: str, judge_version: str, evidence_hash: str
    ) -> StoredJudgement | None:
        row = self._db.execute(
            "SELECT * FROM judgement WHERE judge_id = ? AND judge_version = ? "
            "AND evidence_hash = ? AND status = 'done' ORDER BY judgement_id DESC LIMIT 1",
            (judge_id, judge_version, evidence_hash),
        ).fetchone()
        return None if row is None else _judgement_from_row(row)

    async def save_judgement(self, judgement: StoredJudgement) -> None:
        v = judgement.verdict
        await self._call(
            self._db.execute,
            "INSERT INTO judgement (attempt_id, judge_id, judge_version, evidence_hash, "
            "judge_model, status, verdict_final, verdict_raw, reasks, dropped, verdict_json, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (attempt_id, judge_id, judge_version) DO UPDATE SET "
            "evidence_hash = excluded.evidence_hash, judge_model = excluded.judge_model, "
            "status = excluded.status, verdict_final = excluded.verdict_final, "
            "verdict_raw = excluded.verdict_raw, reasks = excluded.reasks, "
            "dropped = excluded.dropped, verdict_json = excluded.verdict_json, "
            "created_at = excluded.created_at "
            "WHERE judgement.status = 'harness_error'",
            (
                judgement.attempt_id,
                judgement.judge_id,
                judgement.judge_version,
                judgement.evidence_hash,
                judgement.judge_model,
                v.status.value,
                v.verdict_final,
                v.verdict_raw,
                v.reasks,
                len(v.dropped),
                v.model_dump_json(),
                self._now(),
            ),
        )

    async def list_judgements(self, run_id: str) -> list[StoredJudgement]:
        return await self._call(self._list_judgements, run_id)

    def _list_judgements(self, run_id: str) -> list[StoredJudgement]:
        rows = self._db.execute(
            "SELECT j.* FROM judgement j JOIN attempt a USING (attempt_id) "
            "WHERE a.run_id = ? ORDER BY j.attempt_id",
            (run_id,),
        ).fetchall()
        return [_judgement_from_row(r) for r in rows]

    async def save_calibration(self, label: CalibrationLabel) -> None:
        await self._call(
            self._db.execute,
            "INSERT INTO calibration (attempt_id, judge_id, judge_version, labeller, "
            "human_verdict, note) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (attempt_id, judge_id, judge_version, labeller) DO UPDATE SET "
            "human_verdict = excluded.human_verdict, note = excluded.note",
            (
                label.attempt_id,
                label.judge_id,
                label.judge_version,
                label.labeller,
                label.human_verdict,
                label.note,
            ),
        )

    async def list_calibration(self, judge_id: str, judge_version: str) -> list[CalibrationLabel]:
        return await self._call(self._list_calibration, judge_id, judge_version)

    def _list_calibration(self, judge_id: str, judge_version: str) -> list[CalibrationLabel]:
        rows = self._db.execute(
            "SELECT * FROM calibration WHERE judge_id = ? AND judge_version = ? "
            "ORDER BY attempt_id, labeller",
            (judge_id, judge_version),
        ).fetchall()
        return [
            CalibrationLabel(
                attempt_id=r["attempt_id"],
                judge_id=r["judge_id"],
                judge_version=r["judge_version"],
                labeller=r["labeller"],
                human_verdict=r["human_verdict"],
                note=r["note"],
            )
            for r in rows
        ]

    async def get_grade(self, attempt_id: int, grader_version: str) -> Grade | None:
        return await self._call(self._get_grade, attempt_id, grader_version)

    def _get_grade(self, attempt_id: int, grader_version: str) -> Grade | None:
        row = self._db.execute(
            "SELECT outcome, reasons_json FROM grade WHERE attempt_id = ? AND grader_version = ?",
            (attempt_id, grader_version),
        ).fetchone()
        return None if row is None else _grade_from_row(row)

    async def save_grade(self, attempt_id: int, grader_version: str, grade: Grade) -> None:
        await self._call(
            self._db.execute,
            "INSERT INTO grade (attempt_id, grader_version, outcome, reasons_json) "
            "VALUES (?, ?, ?, ?) ON CONFLICT (attempt_id, grader_version) DO UPDATE SET "
            "outcome = excluded.outcome, reasons_json = excluded.reasons_json",
            (attempt_id, grader_version, grade.outcome.value, json.dumps(grade.reasons)),
        )

    async def list_grades(self, run_id: str, grader_version: str) -> list[tuple[Attempt, Grade]]:
        return await self._call(self._list_grades, run_id, grader_version)

    def _list_grades(self, run_id: str, grader_version: str) -> list[tuple[Attempt, Grade]]:
        rows = self._db.execute(
            f"SELECT a.*, g.outcome, g.reasons_json, (SELECT COALESCE(SUM({_EFFECTIVE_COST}), 0) "  # noqa: S608
            "FROM cost_ledger l WHERE l.attempt_id = a.attempt_id) AS cost_micros "
            "FROM attempt a JOIN grade g ON g.attempt_id = a.attempt_id "
            "WHERE a.run_id = ? AND g.grader_version = ? ORDER BY a.attempt_id",
            (run_id, grader_version),
        ).fetchall()
        return [(_attempt_from_row(r), _grade_from_row(r)) for r in rows]


def _attempt_from_row(row: sqlite3.Row) -> Attempt:
    return Attempt(
        attempt_id=row["attempt_id"],
        run_id=row["run_id"],
        case_id=row["case_id"],
        repeat=row["repeat"],
        status=AttemptStatus(row["status"]),
        output=json.loads(row["output_json"]) if row["output_json"] else None,
        error_kind=ErrorKind(row["error_kind"]) if row["error_kind"] else None,
        error_msg=row["error_msg"],
        cost_micros=row["cost_micros"],
        try_count=row["try_count"],
        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
    )


def _judgement_from_row(row: sqlite3.Row) -> StoredJudgement:
    return StoredJudgement(
        attempt_id=row["attempt_id"],
        judge_id=row["judge_id"],
        judge_version=row["judge_version"],
        evidence_hash=row["evidence_hash"],
        judge_model=row["judge_model"],
        verdict=JudgeVerdict.model_validate_json(row["verdict_json"]),
    )


def _grade_from_row(row: sqlite3.Row) -> Grade:
    return Grade.model_validate(
        {"outcome": row["outcome"], "reasons": json.loads(row["reasons_json"])}
    )
