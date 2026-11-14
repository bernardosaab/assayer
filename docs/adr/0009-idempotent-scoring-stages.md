# 0009. Scoring as separate idempotent stages keyed by content-hashed versions

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

Graders change far more often than the agent: a tighter threshold, a new end-state check, a
reworded rubric line. Re-executing the agent to try a new check wastes money. Mixing results
from old and new graders silently corrupts the numbers. When one check changes, exactly that
check should be recomputed, at no model cost.

## Considered options

1. Score inline during execution and store one result per attempt.
2. Separate stages with hand-maintained version numbers.
3. Separate stages, each keyed by a version derived from content.

## Decision

Option 3. `application/scoring.py:ScoreRun` runs three stages. Each reads the store and writes
only what is missing under the current key.

| Stage | Key | Behaviour |
|---|---|---|
| check | `(attempt_id, check_id, check_version)` | deterministic, free; `INSERT ... ON CONFLICT DO NOTHING` (write-once) |
| judge | `(attempt_id, judge_id, judge_version)` | metered model calls; reuse by evidence hash (ADR-0017); overwrite only a `harness_error` row |
| grade | `(attempt_id, grader_version)` | pure; **always recomputed** and upserted |

- `builtin/checks/base.py:SpecCheck.version` is `f"{LOGIC_VERSION}-{content_hash(spec.use,
  args, length=8)}"`. Editing a check's arguments, or bumping its logic version, invalidates
  that check and nothing else.
- `EvalSpec.judge_version` is `content_hash(JUDGE_PROTOCOL_VERSION, judge_spec, judge_model)`.
- `grader_version` is `content_hash(GRADER_VERSION, sorted (check_id, version) over all cases,
  judge_version)`. `_grade` keeps only check rows whose `(check_id, version)` is current.
- The run's `config_hash` covers the system under test, not the graders. A grading change can
  therefore re-score the same run id.

Deterministic sensors come first. Checks run before the judge, for free, and settle whatever
code can settle. That includes end-state checks such as `state_equals`, which scores the write
in the system of record, not the agent's claim about it.

`tests/integration/test_task_pipeline.py::test_changing_one_check_recomputes_only_that_check`
raises the `set_f1` threshold from 0.8 to 0.95 on an existing run. It asserts six new check
results (one per attempt), zero new judgements and zero model calls, and that `amara-okafor`
flips to `fail` (F1 0.91).

## Consequences

**Positive:** re-scoring is free and repeatable. A grader change touches exactly what it should,
and old and new results never mix, because each read filters by the current version.

**Negative / trade-offs:**

- The hash covers a check's *arguments*, not its *code*. A change to check logic needs a manual
  `LOGIC_VERSION` bump, and the same is true of `GRADER_VERSION` and `JUDGE_PROTOCOL_VERSION`.
  If the bump is forgotten, stale rows stay valid, including a stored "check crashed" failure
  that persists after the bug is fixed.
- Rows for superseded versions are never deleted (`retry_errors` clears only the attempts it
  resets), so the tables grow with every grader change.
- `grader_version` is eval-wide. Changing one case's check writes a new grade row for every
  attempt. That is cheap, but reading an older grade requires knowing its version
  (`RunMeta.grader_version`).
- The judge stage loops over attempts one at a time. Judging is not concurrent.
