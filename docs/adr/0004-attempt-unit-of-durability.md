# 0004. The attempt is the unit of durability, claimed before the first model call

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

A run is cases x repeats. One case may make a single model call (extraction) or dozens (a
conversation: agent turns, tool loops, simulator turns). The process can die at any point.
Resuming must skip finished work, never run the same work twice in parallel, and never lose
track of work that was in flight when the process died.

## Considered options

1. Write results at the end of the run.
2. Checkpoint every model call or conversation turn, and resume mid-conversation.
3. One durable row per attempt `(run_id, case_id, repeat)`, claimed before its first call and
   finished or failed as a whole.

## Decision

Option 3.

- `SqliteStore._ensure_attempts` inserts every attempt as `pending` up front
  (`ON CONFLICT (run_id, case_id, repeat) DO NOTHING`, so it is idempotent).
- `Executor.execute` (`application/execute.py`) claims an attempt with `_claim_attempt`, which
  runs `UPDATE attempt SET status = 'in_progress', try_count = try_count + 1 ... WHERE ...
  AND status = 'pending' RETURNING attempt_id` in a `BEGIN IMMEDIATE` transaction. Only one
  claimant can win. Only after the claim does `_attempt` bind the agent and simulator, with the
  `attempt_id` in the `CallContext`, so every ledger row is tied to its attempt.
- The state machine is pure (`domain/attempt.py:TRANSITIONS`): `pending -> in_progress ->
  succeeded | harness_error`, `in_progress -> pending` (reset), and `harness_error -> pending`
  (retry). The store enforces it with `WHERE status = 'in_progress'` guards, and
  `_transition` raises `IllegalTransition` when no row changed.
- The output is written once, by `finish_attempt`, as canonical JSON: a `TaskOutput` or the full
  `Transcript`.
- Reusing a `--run-id` resumes the run. `RunEval` (`application/run_eval.py`) loads or creates
  the run, refusing a different `config_hash`. It then calls `ensure_attempts`,
  `recover_interrupted` (ADR-0005), optionally `retry_errors`, and executes only `pending`
  attempts.

## Consequences

**Positive:** a crash always leaves a trace, in the form of an `in_progress` row plus ledger
rows carrying its `attempt_id`. A resumed run does only the missing work. The exclusive claim
lets two processes share a run without executing an attempt twice. Cost per attempt is a SQL
sum.

**Negative / trade-offs:**

- No per-turn checkpoint. If a conversation crashes on turn 7, the whole conversation is redone,
  and the spend on turns 1-6 is paid again. Because the simulator is itself a model, the redone
  conversation is a new sample, not the old one continued.
- `fail_attempt` stores only `error_kind` and the first 2,000 characters of the message. The
  partial transcript of a failed attempt is discarded (`output_json` is written only on
  success), so debugging a mid-conversation infrastructure failure needs the logs.
- Known gap: when the budget halts a run, `_attempt` returns an attempt to `pending` if
  `calls_made == 0`. `BoundChatModel.calls_made` counts only *completed* calls. An attempt whose
  only call failed after it was sent (worst case kept, ADR-0006) and then hit the budget on a
  retry goes back to `pending` with a charge on the ledger. It does not become a visible
  harness error.
