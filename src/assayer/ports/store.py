"""Persistence ports.

The store is split by consumer (interface segregation): the orchestrator needs runs and
attempts, the cost guard needs only the ledger, and the scoring stages need only scores.
One adapter (SQLite) implements all four, but each collaborator depends on the narrowest one.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol

from assayer.domain.attempt import Attempt
from assayer.domain.base import Model
from assayer.domain.checks import CheckResult
from assayer.domain.errors import ErrorKind
from assayer.domain.fixtures import ConversationFixture, TaskFixture
from assayer.domain.grading import Grade
from assayer.domain.judging import JudgeVerdict, Verdict
from assayer.domain.messages import Purpose, Usage
from assayer.domain.money import Money
from assayer.domain.run import RunMeta, RunStatus


class RecoveryReport(Model):
    reset: int = 0
    interrupted_after_charge: int = 0
    orphaned_reservations: int = 0


class LedgerEntry(Model):
    entry_id: int
    run_id: str
    attempt_id: int | None
    judgement_key: str | None
    purpose: Purpose
    model: str
    status: str
    reserved_micros: int
    actual_micros: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    created_at: datetime


class StoredJudgement(Model):
    attempt_id: int
    judge_id: str
    judge_version: str
    evidence_hash: str
    judge_model: str
    verdict: JudgeVerdict


class CalibrationLabel(Model):
    attempt_id: int
    judge_id: str
    judge_version: str
    labeller: str
    human_verdict: Verdict
    note: str = ""


class RunRepository(Protocol):
    async def create_or_load_run(
        self, meta: RunMeta, cases: Sequence[TaskFixture | ConversationFixture]
    ) -> RunMeta:
        """Create the run, or load it if it exists with the same config hash.

        Raises:
            ConfigError: if the run id exists with a different config hash.
        """
        ...

    async def get_run(self, run_id: str) -> RunMeta | None: ...
    async def list_runs(self) -> list[tuple[RunMeta, RunStatus]]: ...
    async def run_status(self, run_id: str) -> RunStatus: ...
    async def set_run_status(self, run_id: str, status: RunStatus) -> None: ...
    async def load_cases(self, run_id: str) -> list[TaskFixture | ConversationFixture]: ...


class AttemptRepository(Protocol):
    async def ensure_attempts(self, run_id: str, repeats: int) -> None:
        """Create a ``pending`` attempt for every (case, repeat) that has none yet."""
        ...

    async def recover_interrupted(self, run_id: str) -> RecoveryReport: ...

    async def claim_attempt(self, run_id: str, case_id: str, repeat: int) -> Attempt | None:
        """Atomically move a pending attempt to in_progress. None if it is not claimable."""
        ...

    async def finish_attempt(self, attempt_id: int, output: dict[str, Any]) -> None: ...
    async def fail_attempt(self, attempt_id: int, kind: ErrorKind, message: str) -> None: ...
    async def release_attempt(self, attempt_id: int) -> None:
        """Return an in_progress attempt to pending (e.g. halted by budget before spending)."""
        ...

    async def retry_errors(self, run_id: str, *, max_tries: int) -> int: ...
    async def copy_attempts(self, source_run_id: str, target_run_id: str) -> int:
        """Copy finished attempts into another run (replay). Returns the number copied."""
        ...

    async def list_attempts(self, run_id: str) -> list[Attempt]: ...


class CostLedger(Protocol):
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
        """Atomically reserve ``amount`` against the run budget and return the entry id.

        Raises:
            BudgetExceeded: if committed spend plus ``amount`` would exceed the budget.
        """
        ...

    async def settle(self, entry_id: int, actual: Money, usage: Usage | None) -> None: ...
    async def release(self, entry_id: int) -> None: ...
    async def committed(self, run_id: str) -> Money: ...
    async def ledger(self, run_id: str) -> list[LedgerEntry]: ...
    async def raise_budget(self, run_id: str, budget: Money) -> None: ...


class ScoreRepository(Protocol):
    async def get_check_results(self, attempt_id: int) -> list[CheckResult]: ...
    async def save_check_result(self, attempt_id: int, result: CheckResult) -> None: ...

    async def get_judgement(
        self, attempt_id: int, judge_id: str, judge_version: str
    ) -> StoredJudgement | None: ...

    async def find_judgement_by_evidence(
        self, judge_id: str, judge_version: str, evidence_hash: str
    ) -> StoredJudgement | None:
        """Replay cache: a finished judgement of identical evidence under the same judge."""
        ...

    async def save_judgement(self, judgement: StoredJudgement) -> None: ...
    async def list_judgements(self, run_id: str) -> list[StoredJudgement]: ...

    async def save_calibration(self, label: CalibrationLabel) -> None: ...
    async def list_calibration(
        self, judge_id: str, judge_version: str
    ) -> list[CalibrationLabel]: ...

    async def get_grade(self, attempt_id: int, grader_version: str) -> Grade | None: ...
    async def save_grade(self, attempt_id: int, grader_version: str, grade: Grade) -> None: ...
    async def list_grades(
        self, run_id: str, grader_version: str
    ) -> list[tuple[Attempt, Grade]]: ...


class ResultStore(RunRepository, AttemptRepository, CostLedger, ScoreRepository, Protocol):
    """Everything; what the composition root holds."""

    async def close(self) -> None: ...
