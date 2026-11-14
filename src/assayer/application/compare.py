"""Paired comparison of two runs, and the release gate (ADR-0018).

"A prompt change lifts the pass rate from 78% to 82%. Is it real?" Two independent pass rates
cannot answer that on a few dozen cases. This compares the runs **paired on the same
(case, repeat)**:

* discordant pairs: regressions (pass -> fail) and improvements (fail -> pass);
* an exact McNemar test on those pairs;
* both rates with Wilson intervals, and the interval half-width as the smallest change these
  sample sizes can resolve;
* the flipped cases themselves, which are what a person reads next.

The gate (``assayer diff --fail-on-drop``) fails when the candidate's paired pass rate drops by
more than the allowed margin. Harness errors are excluded on both sides.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from assayer.domain.attempt import Attempt
from assayer.domain.base import Model
from assayer.domain.grading import Grade
from assayer.domain.outcome import Outcome
from assayer.domain.stats import Flip, RateSummary, half_width, mcnemar_exact, summarize


class Comparison(Model):
    base_run: str
    candidate_run: str
    base: RateSummary
    candidate: RateSummary
    pairs: int
    regressions: int
    improvements: int
    p_value: float
    paired_delta: float | None = Field(description="Candidate minus base pass rate on pairs.")
    resolvable_change: float | None
    flips: list[Flip]
    unpaired: int

    def regressed(self, max_drop: float) -> bool:
        return self.paired_delta is not None and self.paired_delta < -max_drop


def _key(attempt: Attempt) -> str:
    return f"{attempt.case_id}#{attempt.repeat}"


def compare(
    base_run: str,
    base: Sequence[tuple[Attempt, Grade]],
    candidate_run: str,
    candidate: Sequence[tuple[Attempt, Grade]],
) -> Comparison:
    graded = {Outcome.PASS, Outcome.FAIL}
    b = {_key(a): g.outcome for a, g in base}
    c = {_key(a): g.outcome for a, g in candidate}
    paired = [k for k in sorted(b.keys() & c.keys()) if b[k] in graded and c[k] in graded]
    regressions = sum(b[k] is Outcome.PASS and c[k] is Outcome.FAIL for k in paired)
    improvements = sum(b[k] is Outcome.FAIL and c[k] is Outcome.PASS for k in paired)
    base_passes = sum(b[k] is Outcome.PASS for k in paired)
    delta = (improvements - regressions) / len(paired) if paired else None
    return Comparison(
        base_run=base_run,
        candidate_run=candidate_run,
        base=summarize(b.values()),
        candidate=summarize(c.values()),
        pairs=len(paired),
        regressions=regressions,
        improvements=improvements,
        p_value=mcnemar_exact(regressions, improvements),
        paired_delta=delta,
        resolvable_change=half_width(base_passes, len(paired)) if paired else None,
        flips=[Flip(key=k, base=b[k], candidate=c[k]) for k in paired if b[k] is not c[k]],
        unpaired=len(b.keys() ^ c.keys()),
    )
