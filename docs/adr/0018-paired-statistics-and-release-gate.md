# 0018. Wilson intervals, pass^k, paired McNemar comparisons and a release gate

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

Eval sets are small, and model outputs vary between calls. "78% to 82%" on a few dozen cases is
usually noise, and comparing two independent pass rates needs far more cases than anyone has.
A single pass rate also hides the difference between "can do it" and "does it every time".

## Considered options

1. Point pass rates and a threshold.
2. Independent two-proportion tests between runs.
3. Intervals on every rate, reliability across repeats, and paired comparisons on the same cases.

## Decision

Option 3 (`domain/stats.py`, `application/analytics.py`, `application/compare.py`).

- Every rate excludes harness errors (ADR-0008) and carries a 95% **Wilson** interval, which
  behaves at small n and near 0% or 100%.
- With repeats, **pass^k** (every repeat passed: reliability) and **pass@k** (at least one passed:
  capability) are reported per run.
- `assayer diff` pairs two runs on `(case, repeat)`, counts discordant pairs (regressions and
  improvements), runs an **exact McNemar** test on them, lists the flipped cases, and prints the
  half-width of the base interval as the smallest change the sample can resolve.
- `--fail-on-drop X` exits 1 when the paired pass rate drops by more than X: a release gate CI
  uses (`.github/workflows/ci.yml`). `run --min-pass-rate` is the absolute version.
- Failure causes are counted by check and judge criterion, as the starting point for error
  analysis.

## Consequences

**Positive:** numbers come with their uncertainty; pairing removes case-difficulty noise; the
flipped cases are what a person reads next; the gate is mechanical.

**Negative / trade-offs:** with the example evals' 3-8 cases, only very large changes are
resolvable, and the reports say so rather than hiding it. The gate compares point estimates
against a margin; it does not require significance, because blocking on p-values at this n
would never block anything. Repeats of the same case are not independent, so intervals over
attempts are optimistic when repeats are many.
