# 0006. Reserve-then-settle cost guard with a bytes-as-tokens worst-case bound

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

A run has a hard spend cap (`--budget-usd`). It must hold with concurrent attempts, retries,
crashes and two processes on one run. The real cost of a call is known only after it returns
(`usage`). If the harness checks only after the fact, a burst of concurrent calls overshoots the
cap. The refusal has to happen *before* the request leaves, which requires an upper bound on the
call's cost without running it.

## Considered options

1. Sum actual costs after each call and stop once the budget is exceeded.
2. Estimate prompt tokens with each model's tokenizer.
3. Reserve a sound upper bound before the call, then settle to the actual cost.

## Decision

Option 3, implemented by `MeteredProvider` (`application/provider_middleware.py`) on top of the
`cost_ledger` table.

1. Apply `default_max_tokens` if the request has none. Then compute
   `worst = PriceTable.lookup(model).worst_case(request)`.
2. `ledger.reserve(...)` -> `SqliteStore._reserve`, inside `BEGIN IMMEDIATE`. If committed spend
   (`SUM` of `_EFFECTIVE_COST`) plus `worst` is greater than `budget_micros`, it raises
   `BudgetExceeded` before any I/O. Otherwise it inserts a `reserved` row tagged with run,
   attempt, purpose, model, judgement key and session id.
3. Call the provider, then settle.

The bound is in `domain/pricing.py:prompt_token_upper_bound`. Prompt tokens are taken as the
UTF-8 byte length of `json.dumps([messages, tools, response_schema], ensure_ascii=False)`, plus 8 tokens per message
and 64 per request. Completion tokens are taken as `max_tokens`, which is always sent, so the
provider enforces it. Byte-level BPE tokenizers emit at most one token per byte, so the bound is
sound without shipping a tokenizer.

Settle rules:

| What happened | Ledger row |
|---|---|
| Success with usage | `settled` at `price.cost(usage)`; a warning is logged if it exceeds `worst` |
| Success without usage | `settled` at `worst` |
| `ProviderError` with `request_sent = False` (connect failure, 429, 4xx) | `released` (0) |
| `ProviderError` with `request_sent = True` (timeout, 5xx, mid-request drop, bad body), or any other `Exception` | `settled` at `worst`: the provider may have billed |
| Cancellation (`CancelledError`, crash) | stays `reserved`; the next session marks it `orphaned` (ADR-0005), and it still counts |

When `BudgetExceeded` reaches the `Executor`, it stops claiming new attempts and the run ends
`halted`. Rerunning with a larger `--budget-usd` raises the budget (`raise_budget`) and resumes
(`test_budget_halt_then_resume_with_more_budget`). Every ledger row carries its purpose and
attempt, so cost can be quoted per agent, simulator and judge, capped, and attributed:
`RunAnalytics.cost_by_purpose` and `cost_per_success`.

## Consequences

**Positive:** the cap holds before money is spent, not after. It holds across tasks and
processes (`tests/property/test_ledger_invariants.py`) and across crashes. Each retry is a
separately metered call (`test_each_retry_is_metered_separately`).

**Negative / trade-offs:**

- The bound over-reserves, typically 3-4x on English prompts, plus the full `max_tokens` for the
  completion. Non-English text over-reserves more (two to four UTF-8 bytes per character).
  Near the cap, a run halts while real headroom remains.
- Keeping the worst case on ambiguous failures, and never settling orphans, overstates the
  reported spend.
- A cost above the bound is only logged, not enforced. Soundness has been argued for byte-level
  BPE only.
- Releasing on 429 and 4xx assumes providers do not bill rejected requests.
