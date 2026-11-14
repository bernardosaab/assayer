"""The budget invariant under adversarial interleavings, against the real SQLite ledger.

Invariant: committed spend never exceeds the budget, whatever sequence of reserve / settle /
release / crash-and-recover happens.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import tempfile
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from assayer.adapters.store.sqlite_store import SqliteStore
from assayer.adapters.system.clock import FixedClock
from assayer.domain.errors import BudgetExceeded
from assayer.domain.messages import Purpose
from assayer.domain.money import Money
from tests.support.builders import a_run, a_task

BUDGET = Money(10_000)


class LedgerMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ledger.sqlite"
        self.session = 0
        self.store = self._open()
        self.loop.run_until_complete(
            self.store.create_or_load_run(a_run(budget_micros=BUDGET.micros), [a_task()])
        )
        self.open_entries: list[tuple[int, int]] = []  # (entry_id, reserved)

    def _open(self) -> SqliteStore:
        self.session += 1
        return self.loop.run_until_complete(
            SqliteStore.open(self.path, clock=FixedClock(), session_id=f"s{self.session}")
        )

    def run[T](self, coro: Coroutine[Any, Any, T]) -> T:
        return self.loop.run_until_complete(coro)

    @rule(amount=st.integers(0, 4_000))
    def reserve(self, amount: int) -> None:
        before = self.run(self.store.committed("run-1"))
        try:
            entry = self.run(
                self.store.reserve(
                    run_id="run-1", purpose=Purpose.AGENT, model="m", amount=Money(amount)
                )
            )
        except BudgetExceeded:
            assert before.micros + amount > BUDGET.micros
            return
        self.open_entries.append((entry, amount))

    @precondition(lambda self: self.open_entries)
    @rule(data=st.data())
    def settle(self, data: st.DataObject) -> None:
        entry, reserved = self.open_entries.pop(
            data.draw(st.integers(0, len(self.open_entries) - 1))
        )
        actual = data.draw(st.integers(0, reserved))  # the bound is sound: actual <= reserved
        self.run(self.store.settle(entry, Money(actual), None))

    @precondition(lambda self: self.open_entries)
    @rule()
    def release(self) -> None:
        entry, _ = self.open_entries.pop()
        self.run(self.store.release(entry))

    @rule()
    def crash_and_recover(self) -> None:
        """Abandon open reservations (process died); a new session orphans them."""
        self.run(self.store.close())
        self.store = self._open()
        self.run(self.store.recover_interrupted("run-1"))
        # Orphaned rows may still be settled by a late response; keep them settle-able.

    @invariant()
    def committed_within_budget(self) -> None:
        committed = self.run(self.store.committed("run-1"))
        assert 0 <= committed.micros <= BUDGET.micros

    def teardown(self) -> None:
        self.run(self.store.close())
        self.loop.close()
        self.tmp.cleanup()


TestLedgerMachine = LedgerMachine.TestCase


async def test_concurrent_reservations_never_overspend(tmp_path: Path) -> None:
    store = await SqliteStore.open(tmp_path / "c.sqlite", clock=FixedClock())
    await store.create_or_load_run(a_run(budget_micros=1_000), [a_task()])

    async def one() -> bool:
        try:
            await store.reserve(run_id="run-1", purpose=Purpose.AGENT, model="m", amount=Money(7))
        except BudgetExceeded:
            return False
        return True

    results = await asyncio.gather(*(one() for _ in range(200)))
    assert sum(results) == 1_000 // 7
    assert await store.committed("run-1") == Money(7 * (1_000 // 7))
    await store.close()


def _reserve_in_child(path: str, n: int, queue: mp.Queue) -> None:  # type: ignore[type-arg]
    async def main() -> int:
        store = await SqliteStore.open(path, clock=FixedClock())
        ok = 0
        for _ in range(n):
            try:
                await store.reserve(
                    run_id="run-1", purpose=Purpose.AGENT, model="m", amount=Money(10)
                )
                ok += 1
            except BudgetExceeded:
                pass
        await store.close()
        return ok

    queue.put(asyncio.run(main()))


@pytest.mark.slow
async def test_two_processes_share_one_budget(tmp_path: Path) -> None:
    path = tmp_path / "mp.sqlite"
    store = await SqliteStore.open(path, clock=FixedClock())
    await store.create_or_load_run(a_run(budget_micros=1_000), [a_task()])
    await store.close()

    ctx = mp.get_context("spawn")
    queue: mp.Queue[int] = ctx.Queue()
    procs = [ctx.Process(target=_reserve_in_child, args=(str(path), 80, queue)) for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    successes = queue.get() + queue.get()
    assert successes == 100  # exactly budget / amount across both processes

    store = await SqliteStore.open(path, clock=FixedClock())
    assert await store.committed("run-1") == Money(1_000)
    await store.close()
