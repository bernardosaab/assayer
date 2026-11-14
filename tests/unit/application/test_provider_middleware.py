from __future__ import annotations

import asyncio
import random
from decimal import Decimal

import pytest

from assayer.application.provider_middleware import (
    LimitedProvider,
    MeteredProvider,
    RetryingProvider,
    TimeoutProvider,
)
from assayer.domain.errors import (
    BudgetExceeded,
    ProviderBadRequest,
    ProviderRateLimited,
    ProviderServerError,
    ProviderTimeout,
    ProviderUnavailable,
)
from assayer.domain.messages import CompletionRequest, CompletionResponse, Usage
from assayer.domain.money import Money
from assayer.domain.pricing import ModelPrice, PriceEntry, PriceTable
from tests.support.fakes import InMemoryLedger, StubProvider, a_request

PRICE = ModelPrice(input_per_mtok=Decimal("10"), output_per_mtok=Decimal("10"))
PRICES = PriceTable(entries=[PriceEntry(pattern="*", price=PRICE)])


def _metered(inner: StubProvider, ledger: InMemoryLedger) -> MeteredProvider:
    return MeteredProvider(inner, ledger, PRICES, default_max_tokens=50)


async def test_success_settles_at_actual_usage() -> None:
    ledger = InMemoryLedger()
    await _metered(StubProvider(), ledger).complete(a_request())
    [row] = ledger.rows.values()
    assert row.status == "settled"
    assert row.actual == PRICE.cost(Usage(prompt_tokens=10, completion_tokens=5))
    assert row.actual < row.reserved


async def test_missing_usage_charges_worst_case() -> None:
    ledger = InMemoryLedger()
    await _metered(StubProvider(usage=None), ledger).complete(a_request())
    [row] = ledger.rows.values()
    assert row.actual == row.reserved


async def test_default_max_tokens_is_applied_before_reserving() -> None:
    ledger = InMemoryLedger()
    seen: list[CompletionRequest] = []

    class Spy(StubProvider):
        async def complete(self, request: CompletionRequest) -> CompletionResponse:
            seen.append(request)
            return await super().complete(request)

    await _metered(Spy(), ledger).complete(a_request(max_tokens=None))
    assert seen[0].max_tokens == 50


@pytest.mark.parametrize(
    ("error", "status", "charged"),
    [
        (ProviderUnavailable("refused"), "released", False),
        (ProviderRateLimited("429"), "released", False),
        (ProviderTimeout("slow"), "settled", True),
        (ProviderServerError("500"), "settled", True),
        (RuntimeError("bug"), "settled", True),
    ],
)
async def test_failures_release_or_keep_worst_case(
    error: Exception, status: str, charged: bool
) -> None:
    ledger = InMemoryLedger()
    with pytest.raises(type(error)):
        await _metered(StubProvider(raises=[error]), ledger).complete(a_request())
    [row] = ledger.rows.values()
    assert row.status == status
    assert (row.actual == row.reserved) is charged


async def test_budget_refusal_happens_before_the_call() -> None:
    inner = StubProvider()
    ledger = InMemoryLedger(budget=Money(1))
    with pytest.raises(BudgetExceeded):
        await _metered(inner, ledger).complete(a_request())
    assert inner.calls == 0


async def test_cancellation_leaves_reservation_for_recovery() -> None:
    ledger = InMemoryLedger()
    inner = StubProvider(raises=[asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await _metered(inner, ledger).complete(a_request())
    [row] = ledger.rows.values()
    assert row.status == "reserved"


async def test_metering_requires_run_context() -> None:
    request = a_request()
    request = request.model_copy(
        update={"context": request.context.model_copy(update={"run_id": None})}
    )
    with pytest.raises(ValueError, match="run_id"):
        await _metered(StubProvider(), InMemoryLedger()).complete(request)


# --------------------------------------------------------------------------- retrying


class _Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


async def test_retry_recovers_from_transient_errors() -> None:
    sleeps = _Sleeps()
    inner = StubProvider(raises=[ProviderServerError("500"), ProviderTimeout("t")])
    provider = RetryingProvider(inner, max_retries=3, sleep=sleeps, rng=random.Random(0))
    response = await provider.complete(a_request())
    assert response.message.content == "ok"
    assert inner.calls == 3
    assert len(sleeps.delays) == 2


async def test_retry_gives_up_after_max_and_skips_non_retryable() -> None:
    inner = StubProvider(raises=[ProviderServerError(str(i)) for i in range(5)])
    with pytest.raises(ProviderServerError):
        await RetryingProvider(inner, max_retries=2, sleep=_Sleeps()).complete(a_request())
    assert inner.calls == 3

    inner = StubProvider(raises=[ProviderBadRequest("400")])
    with pytest.raises(ProviderBadRequest):
        await RetryingProvider(inner, sleep=_Sleeps()).complete(a_request())
    assert inner.calls == 1


async def test_retry_honours_retry_after() -> None:
    sleeps = _Sleeps()
    inner = StubProvider(raises=[ProviderRateLimited("429", retry_after_s=7.0)])
    await RetryingProvider(inner, sleep=sleeps).complete(a_request())
    assert sleeps.delays == [7.0]


async def test_each_retry_is_metered_separately() -> None:
    ledger = InMemoryLedger()
    inner = StubProvider(raises=[ProviderTimeout("t")])
    provider = RetryingProvider(_metered(inner, ledger), sleep=_Sleeps())
    await provider.complete(a_request())
    assert [r.status for r in ledger.rows.values()] == ["settled", "settled"]


# --------------------------------------------------------------------------- timeout / limit


class _Slow(StubProvider):
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        await asyncio.sleep(1)
        return await super().complete(request)


async def test_timeout_maps_to_provider_timeout() -> None:
    with pytest.raises(ProviderTimeout):
        await TimeoutProvider(_Slow(), timeout_s=0.01).complete(a_request())


async def test_limited_provider_caps_concurrency() -> None:
    in_flight = peak = 0

    class Tracking(StubProvider):
        async def complete(self, request: CompletionRequest) -> CompletionResponse:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.001)
            in_flight -= 1
            return await super().complete(request)

    provider = LimitedProvider(Tracking(), max_concurrency=2)
    await asyncio.gather(*(provider.complete(a_request()) for _ in range(10)))
    assert peak == 2
    assert provider.name == "stub"
