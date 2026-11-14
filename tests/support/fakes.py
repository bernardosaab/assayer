"""Hand-written test doubles for ports (no mocking library needed)."""

from __future__ import annotations

from dataclasses import dataclass, field

from assayer.domain.errors import BudgetExceeded
from assayer.domain.messages import (
    CallContext,
    CompletionRequest,
    CompletionResponse,
    Message,
    Purpose,
    Usage,
)
from assayer.domain.money import Money
from assayer.ports.store import LedgerEntry


@dataclass
class _Row:
    reserved: Money
    status: str = "reserved"
    actual: Money | None = None


@dataclass
class InMemoryLedger:
    """A minimal ``CostLedger`` for unit-testing the cost middleware in isolation."""

    budget: Money = field(default_factory=lambda: Money(1_000_000))
    rows: dict[int, _Row] = field(default_factory=dict)

    def _effective(self, row: _Row) -> Money:
        if row.status == "settled":
            assert row.actual is not None
            return row.actual
        if row.status == "released":
            return Money.zero()
        return row.reserved

    async def committed(self, run_id: str) -> Money:
        return sum((self._effective(r) for r in self.rows.values()), Money.zero())

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
        committed = await self.committed(run_id)
        if committed + amount > self.budget:
            raise BudgetExceeded(committed=committed, requested=amount, budget=self.budget)
        entry = len(self.rows) + 1
        self.rows[entry] = _Row(reserved=amount)
        return entry

    async def settle(self, entry_id: int, actual: Money, usage: Usage | None) -> None:
        row = self.rows[entry_id]
        if row.status == "reserved":
            row.status, row.actual = "settled", actual

    async def release(self, entry_id: int) -> None:
        row = self.rows[entry_id]
        if row.status == "reserved":
            row.status, row.actual = "released", Money.zero()

    async def ledger(self, run_id: str) -> list[LedgerEntry]:
        raise NotImplementedError

    async def raise_budget(self, run_id: str, budget: Money) -> None:
        self.budget = budget


@dataclass
class StubProvider:
    """Returns a fixed reply, or raises the queued exceptions first."""

    reply: str = "ok"
    usage: Usage | None = field(
        default_factory=lambda: Usage(prompt_tokens=10, completion_tokens=5)
    )
    raises: list[BaseException] = field(default_factory=list)
    calls: int = 0
    name: str = "stub"

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls += 1
        if self.raises:
            raise self.raises.pop(0)
        return CompletionResponse(
            message=Message.assistant(self.reply), model=request.model, usage=self.usage
        )


def a_request(
    *, model: str = "paid/model", max_tokens: int | None = 100, text: str = "hello"
) -> CompletionRequest:
    return CompletionRequest(
        model=model,
        messages=(Message.user(text),),
        context=CallContext(purpose=Purpose.AGENT, run_id="run-1", attempt_id=1),
        max_tokens=max_tokens,
    )
