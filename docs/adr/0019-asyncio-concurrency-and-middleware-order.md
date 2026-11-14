# 0019. asyncio concurrency, provider middleware order, and a circuit breaker

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

Eval runs are I/O-bound: most time is spent waiting for models. Endpoints differ in how much
concurrency they tolerate (a laptop running Ollama wants one request at a time; a hosted API
wants several), they rate-limit, and sometimes they are simply down. Retries, limits, timeouts
and metering must compose without double-counting or leaking spend.

## Considered options

1. Threads or processes per attempt.
2. asyncio, with each concern inside one provider class.
3. asyncio, with each concern as a decorator around the provider.

## Decision

Option 3.

- `Executor` (`application/execute.py`) runs pending attempts in an `asyncio.TaskGroup` under a
  global semaphore (`--concurrency`). Workers check a halt event before claiming more work.
- The composition root (`cli/container.py:_bind`) wraps every role's provider as
  `Retrying(Limited(Metered(Timeout(provider))))` (`application/provider_middleware.py`):
  - **Retrying outermost**: each retry is a fresh metered call with its own reservation, so a
    retry can never spend money the ledger does not know about. Backoff is exponential with full
    jitter and honours `Retry-After`.
  - **Limited outside Metered**: a call waiting for an endpoint slot holds no reservation, so a
    queue does not eat the budget.
  - **Timeout innermost**: a timeout surfaces as a `ProviderTimeout` the meter accounts for
    (request sent, worst case kept).
- A **circuit breaker** halts the run after N consecutive harness errors (default 5): the
  endpoint is down, and recording the same failure for every remaining case is waste.
- SQLite calls go through the store's single writer thread (ADR-0003), so blocking I/O never
  runs on the event loop.

## Consequences

**Positive:** each concern is small and unit-tested in isolation
(`tests/unit/application/test_provider_middleware.py`); per-endpoint limits suit local models;
an outage stops a run quickly and visibly.

**Negative / trade-offs:** the order is load-bearing and only documented here and in the module
docstring; a new decorator in the wrong place could reintroduce leaked spend. The consecutive
error counter is approximate under concurrency. Long CPU-bound work in a check or tool would
block the loop; built-in checks are cheap, but user code is not policed.
