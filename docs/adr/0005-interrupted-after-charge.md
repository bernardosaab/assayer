# 0005. Interrupted attempts that spent money become a visible harness error

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

On resume, any attempt still `in_progress` was interrupted by a crash, Ctrl-C or `kill -9`.
Re-running it silently is tempting. But if it already made billed calls, a silent retry pays
twice and hides the crash from the person reading the results. If it made no billed call,
redoing it costs nothing extra and should just happen. Either way, the money reserved for the
call in flight must not drop out of the budget.

## Considered options

1. Reset every interrupted attempt to `pending` (automatic retry).
2. Mark every interrupted attempt as a harness error.
3. Decide per attempt by committed spend: reset if nothing was charged, otherwise record an
   `interrupted_after_charge` harness error that needs an explicit retry.

## Decision

Option 3.

- `domain/attempt.py:recover(status, charged=...)` is a pure function. An `in_progress` attempt
  becomes `pending` when it was not charged, and `harness_error` with
  `ErrorKind.INTERRUPTED_AFTER_CHARGE` when it was.
- `SqliteStore._recover_interrupted` runs in one `BEGIN IMMEDIATE` transaction. First it marks
  ledger rows still `reserved` by *another* session (`session_id != ?`) as `orphaned`. Then,
  for each `in_progress` attempt, it sums `_EFFECTIVE_COST` over the attempt's ledger rows
  (`settled` -> actual, `released` -> 0, `reserved`/`orphaned` -> reserved amount) and calls
  `recover(charged=cost > 0)`. The error message reads "interrupted after spending; rerun with
  --retry-errors". It returns a `RecoveryReport(reset, interrupted_after_charge,
  orphaned_reservations)`.
- `assayer run ... --retry-errors` calls `_retry_errors(max_tries)`. Harness-error attempts
  with `try_count < max_tries` (`[defaults] max_tries = 3`) go back to `pending`, and their
  `grade`, `check_result`, `judgement` and `calibration` rows are deleted as stale.

`tests/e2e/test_crash_resume.py` proves this end to end with the real CLI in a subprocess.
`FaultInjectingProvider` calls `os._exit(137)` on the third model call, after the cost guard has
reserved money for it. The test then asserts two `succeeded`, one `in_progress` and one
`reserved` ledger row. After a plain resume it finds one `interrupted_after_charge` attempt and
the reservation now `orphaned`. A third run with `--retry-errors` leaves every attempt
`succeeded`.

Skips are data: an interrupted attempt stays in the results as a counted harness error by kind,
instead of disappearing into a retry.

## Consequences

**Positive:** crashes show up in the summary (`harness_error_kinds`), and nothing is silently
paid for twice. Orphaned reservations keep counting against the budget, so the cap holds across
crashes.

**Negative / trade-offs:**

- Retrying needs an explicit `--retry-errors`. A resumed run ends `complete` with the
  interrupted attempt excluded from the rate (ADR-0008), so someone has to notice the error
  count and rerun.
- An orphaned reservation is never settled, because nobody knows what the dead call actually
  cost. It stays counted at the worst case, so reported cost overstates true spend and budget
  headroom shrinks.
- "Charged" means priced spend above zero, not "a request was sent". Interrupted attempts on
  free models (`ollama/*` is priced at 0) are always reset and redone silently.
- After `max_tries` claims, an attempt is no longer retried and stays a harness error.
- The e2e test covers only `--concurrency 1`.
