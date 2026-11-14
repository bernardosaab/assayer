"""The execute stage: run every pending attempt through the system under test.

This is the only stage that calls the agent or the simulator. Each attempt is claimed (row
``in_progress``) before its first model call and finished or failed when it returns, so a crash
at any point leaves an accurate record (ADR-0004).

Halting, the "stop" paths:

* ``BudgetExceeded`` (the cost guard refused a call): stop claiming new work. An attempt that
  had not made a call yet goes back to ``pending``; one that had becomes a harness error.
* Circuit breaker: N consecutive harness errors means the service is down. Stop rather than
  burn through the eval recording identical failures.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

from assayer.application.conversation import run_conversation
from assayer.application.events import (
    AttemptFinished,
    AttemptStarted,
    Halted,
    NullObserver,
    RunObserver,
    StageStarted,
    safe_notify,
)
from assayer.application.harness import Harness
from assayer.domain.attempt import Attempt, AttemptStatus
from assayer.domain.errors import BudgetExceeded, ConfigError, ErrorKind, HarnessError
from assayer.domain.fixtures import ConversationFixture, TaskFixture
from assayer.domain.messages import Purpose
from assayer.domain.outputs import TaskOutput, Transcript
from assayer.ports.store import AttemptRepository, RunRepository
from assayer.ports.sut import ChatModel

log = logging.getLogger(__name__)


class _AttemptsAndRuns(AttemptRepository, RunRepository, Protocol):
    pass


@dataclass
class ExecutionReport:
    executed: int = 0
    succeeded: int = 0
    harness_errors: int = 0
    released: int = 0
    halted_reason: str | None = None
    errors_by_kind: dict[str, int] = field(default_factory=dict)


class Executor:
    def __init__(
        self,
        store: _AttemptsAndRuns,
        harness: Harness,
        *,
        concurrency: int = 4,
        circuit_breaker: int = 5,
        observer: RunObserver | None = None,
    ) -> None:
        self._store = store
        self._harness = harness
        self._concurrency = concurrency
        self._breaker = circuit_breaker
        self._observer: RunObserver = observer or NullObserver()

    async def execute(self, run_id: str) -> ExecutionReport:
        cases = {case.id: case for case in await self._store.load_cases(run_id)}
        pending = [
            a for a in await self._store.list_attempts(run_id) if a.status is AttemptStatus.PENDING
        ]
        report = ExecutionReport()
        halt = asyncio.Event()
        semaphore = asyncio.Semaphore(self._concurrency)
        consecutive_errors = 0
        safe_notify(self._observer, StageStarted("execute", len(pending)))

        async def one(attempt: Attempt) -> None:
            nonlocal consecutive_errors
            async with semaphore:
                if halt.is_set():
                    return
                claimed = await self._store.claim_attempt(
                    attempt.run_id, attempt.case_id, attempt.repeat
                )
                if claimed is None:
                    return
                report.executed += 1
                safe_notify(self._observer, AttemptStarted(claimed.case_id, claimed.repeat))
                started = time.perf_counter()
                outcome = await self._attempt(claimed, cases[claimed.case_id])
                safe_notify(
                    self._observer,
                    AttemptFinished(
                        claimed.case_id,
                        claimed.repeat,
                        ok=outcome is None,
                        error=None if outcome is None else f"{outcome.kind.value}: {outcome}",
                        duration_ms=(time.perf_counter() - started) * 1000,
                    ),
                )
                if outcome is None:
                    report.succeeded += 1
                    consecutive_errors = 0
                    return
                if isinstance(outcome, BudgetExceeded):
                    report.halted_reason = f"budget exhausted ({outcome})"
                    halt.set()
                    return
                report.harness_errors += 1
                report.errors_by_kind[outcome.kind.value] = (
                    report.errors_by_kind.get(outcome.kind.value, 0) + 1
                )
                consecutive_errors += 1
                if consecutive_errors >= self._breaker and not halt.is_set():
                    report.halted_reason = (
                        f"circuit breaker: {consecutive_errors} consecutive harness errors "
                        f"(last: {outcome.kind.value}); is the model endpoint up?"
                    )
                    halt.set()

        async with asyncio.TaskGroup() as group:
            for attempt in pending:
                group.create_task(one(attempt))
        if report.halted_reason:
            safe_notify(self._observer, Halted(report.halted_reason))

        if report.halted_reason and "budget" in report.halted_reason:
            # Attempts released after the budget halt are counted for the summary.
            report.released = sum(
                a.status is AttemptStatus.PENDING for a in await self._store.list_attempts(run_id)
            )
        return report

    async def _attempt(
        self, attempt: Attempt, case: TaskFixture | ConversationFixture
    ) -> HarnessError | None:
        """Run one claimed attempt and record its result. Returns the harness error, if any."""
        h = self._harness
        agent_llm = h.agent.bind(
            h.context(Purpose.AGENT, attempt.run_id, attempt.case_id, attempt.attempt_id)
        )
        sim_llm = (
            h.simulator.bind(
                h.context(Purpose.SIMULATOR, attempt.run_id, attempt.case_id, attempt.attempt_id)
            )
            if h.simulator
            else None
        )
        try:
            output = await self._run_sut(case, agent_llm, sim_llm)
        except BudgetExceeded as exc:
            spent_calls = agent_llm.calls_made + (sim_llm.calls_made if sim_llm else 0)
            if spent_calls == 0:
                await self._store.release_attempt(attempt.attempt_id)
            else:
                await self._store.fail_attempt(attempt.attempt_id, exc.kind, str(exc))
            return exc
        except HarnessError as exc:
            await self._store.fail_attempt(attempt.attempt_id, exc.kind, str(exc))
            return exc
        except Exception as exc:  # a bug in SUT or harness code: visible, never silent
            log.exception("attempt %s crashed", attempt.attempt_id)
            error = HarnessError(f"{type(exc).__name__}: {exc}", kind=ErrorKind.INTERNAL)
            await self._store.fail_attempt(attempt.attempt_id, error.kind, str(error))
            return error
        await self._store.finish_attempt(attempt.attempt_id, output.model_dump(mode="json"))
        return None

    async def _run_sut(
        self,
        case: TaskFixture | ConversationFixture,
        agent_llm: ChatModel,
        sim_llm: ChatModel | None,
    ) -> TaskOutput | Transcript:
        h = self._harness
        if isinstance(case, TaskFixture):
            if h.task_runner is None:
                raise ConfigError("task case but no task runner configured")
            return await h.task_runner.run(case, agent_llm)
        if (
            h.conversational_agent is None
            or h.user_simulator is None
            or h.environment_factory is None
            or sim_llm is None
            or h.spec.agent is None
        ):
            raise ConfigError("conversation case needs agent, simulator and environment")
        return await run_conversation(
            case,
            agent=h.conversational_agent,
            simulator=h.user_simulator,
            environment=h.environment_factory.create(case),
            agent_llm=agent_llm,
            simulator_llm=sim_llm,
            system_prompt=h.spec.agent.system_prompt,
            on_turn=lambda event: safe_notify(self._observer, event),
        )
