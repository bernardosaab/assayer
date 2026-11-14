# 0017. Replay re-scores stored outputs with zero agent calls

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

Graders change as much as agents do: criteria drift once people read outputs, and a judge can be
wrong in ways that look like model regressions. When a pass rate moves, you need to know whether
the agent changed or the grading did, without paying to run the agent again.

## Considered options

1. Re-run everything after any grader change.
2. Re-score the original run in place.
3. Copy the outputs into a new, linked run and score that with the current graders.

## Decision

Option 3: `ReplayRun` (`application/replay.py`), `assayer replay RUN --into NEW`.

- The new run records `replay_of`, the current judge and grader versions, and its own budget.
  `copy_attempts` copies finished outputs, timings and errors; the agent is never called.
- Checks always recompute (deterministic, free).
- Judgements are reused through `find_judgement_by_evidence`: same judge id, same judge version,
  same evidence hash. If only checks changed, replay makes **zero** model calls
  (`tests/integration/test_replay_and_report.py`).
- If the rubric or judge model changed, judgements are not in the cache. By default they are
  skipped and a note says so (`ScoreRun(judge_cache_only=True)`); `--rejudge` allows the judge
  calls, metered against the replay run's budget.
- `assayer diff RUN NEW` then shows exactly which cases flipped because of grading.

## Consequences

**Positive:** grading changes are isolated and cheap to measure; the original run is never
mutated, so its numbers stay reproducible.

**Negative / trade-offs:** the cache assumes a judge is deterministic for identical evidence
under the same version. With temperature above zero that is not strictly true, so cached reuse
hides judge variance; the judge model and protocol version are part of the key to limit this.
Replay cannot evaluate a *new* agent: changed prompts need a real run.
