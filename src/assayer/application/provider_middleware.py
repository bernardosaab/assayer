"""Provider decorators (Decorator pattern). Each adds one cross-cutting concern.

The order is significant. The composition root builds::

    Retrying( Limited( Metered( Timeout( <real provider> ))))

* ``Retrying`` is outermost, so each retry is a *new* metered call with its own reservation;
  a retried call can never spend money the ledger does not know about.
* ``Limited`` sits outside ``Metered`` so calls waiting for a slot do not hold reservations.
* ``Timeout`` is innermost, so a timeout is a provider error that ``Metered`` accounts for
  (the request was sent, so the reservation is kept).
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable

from assayer.domain.errors import ProviderError, ProviderRateLimited, ProviderTimeout
from assayer.domain.messages import CompletionRequest, CompletionResponse
from assayer.domain.pricing import PriceTable
from assayer.ports.provider import ModelProvider
from assayer.ports.store import CostLedger

log = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]


class MeteredProvider:
    """Reserve the worst case before the call, settle the real cost after (ADR-0006).

    * Success: settle at the priced usage. Missing usage settles at the reserved worst case.
    * Provider error whose request was never sent (refused connection, 429): release.
    * Any other failure: the provider may have billed us, so settle at the worst case.
    * Cancellation (Ctrl-C, crash): the row stays ``reserved``; the next resume marks it
      ``orphaned`` and keeps counting it.
    """

    def __init__(
        self,
        inner: ModelProvider,
        ledger: CostLedger,
        prices: PriceTable,
        *,
        default_max_tokens: int = 1024,
    ) -> None:
        self._inner = inner
        self._ledger = ledger
        self._prices = prices
        self._default_max_tokens = default_max_tokens

    @property
    def name(self) -> str:
        return self._inner.name

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        ctx = request.context
        if ctx.run_id is None:
            raise ValueError("metered calls need a run_id in their CallContext")
        if request.max_tokens is None:
            request = request.model_copy(update={"max_tokens": self._default_max_tokens})
        price = self._prices.lookup(request.model)
        worst = price.worst_case(request)
        entry = await self._ledger.reserve(
            run_id=ctx.run_id,
            purpose=ctx.purpose,
            model=request.model,
            amount=worst,
            attempt_id=ctx.attempt_id,
            judgement_key=ctx.judgement_key,
        )
        try:
            response = await self._inner.complete(request)
        except ProviderError as exc:
            if exc.request_sent:
                await self._ledger.settle(entry, worst, None)
            else:
                await self._ledger.release(entry)
            raise
        except Exception:
            await self._ledger.settle(entry, worst, None)
            raise

        if response.usage is None:
            log.warning("provider returned no usage for %s; charging worst case", request.model)
            actual = worst
        else:
            actual = price.cost(response.usage)
            if actual > worst:
                log.warning(
                    "cost bound violated for %s: actual %s > reserved %s",
                    request.model,
                    actual,
                    worst,
                )
        await self._ledger.settle(entry, actual, response.usage)
        return response


class TimeoutProvider:
    def __init__(self, inner: ModelProvider, *, timeout_s: float) -> None:
        self._inner = inner
        self._timeout_s = timeout_s

    @property
    def name(self) -> str:
        return self._inner.name

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        try:
            async with asyncio.timeout(self._timeout_s):
                return await self._inner.complete(request)
        except TimeoutError as exc:
            raise ProviderTimeout(f"no response within {self._timeout_s}s") from exc


class LimitedProvider:
    """Caps concurrent in-flight calls to one endpoint (e.g. 1 for a laptop running Ollama)."""

    def __init__(self, inner: ModelProvider, *, max_concurrency: int) -> None:
        self._inner = inner
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @property
    def name(self) -> str:
        return self._inner.name

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        async with self._semaphore:
            return await self._inner.complete(request)


class RetryingProvider:
    """Retry retryable provider errors with capped exponential backoff and full jitter."""

    def __init__(
        self,
        inner: ModelProvider,
        *,
        max_retries: int = 3,
        base_delay_s: float = 0.5,
        max_delay_s: float = 20.0,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._inner = inner
        self._max_retries = max_retries
        self._base = base_delay_s
        self._cap = max_delay_s
        self._sleep = sleep
        self._rng = rng or random.Random()  # noqa: S311 - jitter, not crypto

    @property
    def name(self) -> str:
        return self._inner.name

    def _delay(self, attempt: int, exc: ProviderError) -> float:
        if isinstance(exc, ProviderRateLimited) and exc.retry_after_s is not None:
            return min(exc.retry_after_s, self._cap)
        return self._rng.uniform(0, min(self._cap, self._base * 2**attempt))

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        attempt = 0
        while True:
            try:
                return await self._inner.complete(request)
            except ProviderError as exc:
                if not exc.retryable or attempt >= self._max_retries:
                    raise
                delay = self._delay(attempt, exc)
                log.info("retrying %s after %s (%.2fs)", self.name, exc.kind.value, delay)
                attempt += 1
                await self._sleep(delay)
