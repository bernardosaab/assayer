"""A budget halt *in the middle of* an attempt (after money was spent).

The attempt that already spent is a harness error (its money is gone, it is excluded from rates),
the run halts for the budget, and the halt is never reported as a circuit-breaker outage, even
with a breaker threshold of 1.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from assayer.adapters.store.sqlite_store import SqliteStore
from assayer.adapters.system.clock import FixedClock
from assayer.adapters.system.git import FakeVcs
from assayer.application.chat_model import RoleSettings
from assayer.application.harness import Harness, RoleBinding
from assayer.application.provider_middleware import MeteredProvider
from assayer.application.run_eval import RunEval, RunOptions
from assayer.domain.attempt import AttemptStatus
from assayer.domain.errors import ErrorKind
from assayer.domain.fixtures import TaskFixture
from assayer.domain.messages import CallContext, CompletionRequest, Message, Purpose
from assayer.domain.money import Money
from assayer.domain.outputs import TaskOutput
from assayer.domain.pricing import ModelPrice, PriceEntry, PriceTable
from assayer.domain.run import RunStatus
from assayer.ports.sut import ChatModel
from tests.support.fakes import StubProvider
from tests.support.harness import load_spec

PRICE = ModelPrice(input_per_mtok=Decimal("10"), output_per_mtok=Decimal("10"))
MAX_TOKENS = 100


class TwoCallTask:
    """Spends on its first call, then needs a second one."""

    async def run(self, fixture: TaskFixture, llm: ChatModel) -> TaskOutput:
        await llm.chat([Message.user("step one")])
        await llm.chat([Message.user("step two")])
        return TaskOutput(raw="{}", parsed={})


def _one_call_worst_case() -> Money:
    request = CompletionRequest(
        model="paid/m",
        messages=(Message.user("step one"),),
        context=CallContext(purpose=Purpose.AGENT),
        max_tokens=MAX_TOKENS,
    )
    return PRICE.worst_case(request)


async def test_budget_exhausted_mid_attempt(tmp_path: Path) -> None:
    store = await SqliteStore.open(tmp_path / "r.sqlite", clock=FixedClock())
    spec = load_spec("resume_extraction")
    metered = MeteredProvider(
        StubProvider(usage=None),  # no usage reported: each call settles at its worst case
        store,
        PriceTable(entries=[PriceEntry(pattern="*", price=PRICE)]),
        default_max_tokens=MAX_TOKENS,
    )
    harness = Harness(
        spec=spec,
        agent=RoleBinding(metered, RoleSettings(model="paid/m", max_tokens=MAX_TOKENS)),
        task_runner=TwoCallTask(),
    )
    budget = Money(_one_call_worst_case().micros * 3 // 2)  # room for one call, not two

    result = await RunEval(store, harness, clock=FixedClock(), vcs=FakeVcs())(
        RunOptions(run_id="r1", budget=budget, concurrency=1, circuit_breaker=1, score=False)
    )
    attempts = await store.list_attempts("r1")
    committed = await store.committed("r1")
    await store.close()

    assert result.status is RunStatus.HALTED
    assert result.execution.halted_reason is not None
    assert result.execution.halted_reason.startswith("budget exhausted")
    assert "circuit breaker" not in result.execution.halted_reason
    assert result.execution.errors_by_kind == {ErrorKind.BUDGET_EXCEEDED.value: 1}
    [spent] = [a for a in attempts if a.status is AttemptStatus.HARNESS_ERROR]
    assert spent.error_kind is ErrorKind.BUDGET_EXCEEDED
    assert spent.cost_micros > 0  # the money it spent is recorded, not forgotten
    assert sum(a.status is AttemptStatus.PENDING for a in attempts) == len(attempts) - 1
    assert committed <= budget
