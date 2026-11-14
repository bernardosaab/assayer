# 0007. Money as integer micro-dollars; unpriced models fail closed

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

A single call can cost $0.000015, and a run sums thousands of them. Binary floats accumulate
error, and the budget comparison in ADR-0006 must be exact. A model missing from the price table
is a second hazard. If it is treated as free, a typo in a model id quietly removes the budget.

## Considered options

1. Float dollars.
2. `Decimal` everywhere, including storage.
3. Integer micro-dollars in memory and in SQL, with `Decimal` only when converting prices.

For unknown models: (a) price them at zero with a warning, or (b) refuse them unless the user
explicitly opts out.

## Decision

Option 3 with (b).

- `domain/money.py:Money` is a frozen dataclass holding an `int` count of micro-dollars. It
  rejects non-`int` values and `bool`. `Money.from_usd` rounds with `ROUND_CEILING`, so a cost
  is never under-counted. The SQL columns (`budget_micros`, `reserved_micros`,
  `actual_micros`) are `INTEGER` with `CHECK (... >= 0)`.
- `ModelPrice` (`domain/pricing.py`) holds `Decimal` USD per million tokens and converts each
  cost to `Money` once.
- `PriceTable.lookup` tries ordered `fnmatchcase` globs from `prices.yaml`, and the first match
  wins. If nothing matches, it raises `ConfigError` ("unpriced models are refused so a budget can
  never be bypassed"). Order matters: `groq/llama-3.1-8b-instant` is listed before `groq/*`.
  Local `ollama/*` is priced at 0. The `fake/*` models have made-up non-zero prices, so offline
  runs exercise the cost guard.
- `--allow-unpriced` (`cli/container.py:_prices`) appends a catch-all `*` pattern priced at zero
  and logs "unknown models are treated as free; budgets cannot protect them". The fake profile
  turns this on automatically only when no `prices.yaml` exists.

## Consequences

**Positive:** budget arithmetic is exact and reproducible, and rounding errs toward overstating
cost. A misspelled model id cannot spend money unmetered.

**Negative / trade-offs:**

- Prices are maintained by hand in `prices.yaml`. A stale price makes the cap wrong in either
  direction. The file can only say "check hosted prices before use".
- The price model is input/output per million tokens only. Cached-input, batch and per-request
  pricing cannot be expressed.
- Ceiling rounding adds up to one micro-dollar per call.
- The refusal happens at the first metered call, not at startup. `lookup` runs inside
  `MeteredProvider.complete`. Its `ConfigError` is not a `HarnessError`, so each claimed attempt
  becomes an `internal` harness error until the circuit breaker halts the run, rather than
  failing before any attempt is claimed.
- `--allow-unpriced` is all-or-nothing: every unknown model becomes free, not just the one you
  meant.
