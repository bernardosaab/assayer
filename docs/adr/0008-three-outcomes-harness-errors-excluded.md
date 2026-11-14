# 0008. Three outcomes: pass, fail, harness_error; harness errors never count toward a rate

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

An eval number has to measure the agent, not the infrastructure around it. If a 503 is scored as
a fail, an outage looks like a regression. The opposite mistake is just as bad: if the agent's
own misbehaviour (unparseable JSON, an endless tool loop) is classed as "error" and excluded,
the rate flatters the agent. The line between the two has to be drawn once, in code.

## Considered options

1. Binary pass/fail; any exception is a fail.
2. pass/fail/error, where any exception is an error.
3. pass/fail/harness_error, with a strict rule: infrastructure failure is `harness_error`, and
   anything the agent produced is graded.

## Decision

Option 3.

- `domain/outcome.py:Outcome` has `PASS`, `FAIL` and `HARNESS_ERROR`. `domain/errors.py` defines
  the taxonomy. `HarnessError` subclasses carry an `ErrorKind`: the provider kinds (with
  `retryable` / `request_sent`), `store_locked`, `budget_exceeded`,
  `interrupted_after_charge`, `judge_inconsistent`, `judge_unparseable` and `internal`.
- Agent misbehaviour stays in the output and is graded:
  - `ExtractionTask` sets `TaskOutput.agent_error` for non-JSON output or missing required
    fields.
  - `ToolAgent` stops at `max_tool_steps` tool calls per turn. The cap lives in code, not in the
    prompt. `run_conversation` records `TerminationReason.AGENT_TOOL_LOOP` and
    `Transcript.agent_error`.
  - `openai_compat` keeps unparseable tool arguments as `{"_raw": ...}`, and the agent receives
    an error tool result.
- `domain/grading.py:grade` applies a precedence order. An attempt or judge harness error gives
  `harness_error`. Otherwise any agent error, failed check or verified judge finding gives `fail`,
  with every reason listed. Otherwise the outcome is `pass`.
- `domain/stats.py:RateSummary.graded` is `passed + failed`. Wilson intervals, pass^k, flips and
  paired comparisons (ADR-0018) all ignore harness errors.
- In `Executor._attempt`, any non-harness exception (a bug in SUT or harness code) becomes an
  `internal` harness error and is logged. It is never swallowed.
- Circuit breaker: after `circuit_breaker` consecutive harness errors (default 5), the run halts.
  A succeeded attempt resets the counter, including one graded `fail`, so a bad agent never
  trips the breaker and a dead endpoint always does
  (`test_circuit_breaker_stops_a_dead_endpoint`: `graded == 0`, `harness_errors == 3`).

Skips are data: harness errors are excluded from the rate but never dropped. They are counted
and broken down by kind (`RunAnalytics.harness_error_kinds`) next to the rate.

## Consequences

**Positive:** an outage cannot pose as a regression, and a misbehaving agent cannot hide behind
"error". Every number has a defined denominator.

**Negative / trade-offs:**

- Excluding harness errors biases the rate when errors correlate with difficulty. If long
  conversations time out more often, the hard cases leave the denominator and the rate rises.
  The only guard is that the error count is shown beside the rate.
- The classification is a judgement call. A slow agent that hits `timeout_s` is recorded as
  `provider_timeout` (infrastructure), although latency is arguably agent behaviour.
- The error paths are not symmetric. An exception thrown by environment tool code is an
  `internal` harness error, because `_invoke` catches only `ToolError`. A check that crashes
  during scoring is stored as a *failed* check (`ScoreRun._check`), not a harness error.
- `ErrorKind` grows with each new failure mode, and adapters must classify each one correctly.
