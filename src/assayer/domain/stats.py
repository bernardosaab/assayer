"""Rates, confidence intervals and run-to-run comparison (ADR-0018).

Eval sets are small (tens of cases), so a point pass rate alone is misleading. Each rate has a
Wilson score interval, and comparisons list the *flipped cases*, which are what a person
actually reads when hill-climbing.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping

from assayer.domain.base import Model
from assayer.domain.outcome import Outcome


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. ``(0, 1)`` when ``n == 0``."""
    if n < 0 or not 0 <= successes <= n:
        raise ValueError("require 0 <= successes <= n")
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


class RateSummary(Model):
    passed: int
    failed: int
    harness_errors: int

    @property
    def graded(self) -> int:
        """Attempts that count toward the rate. Harness errors are excluded (ADR-0008)."""
        return self.passed + self.failed

    @property
    def pass_rate(self) -> float | None:
        return self.passed / self.graded if self.graded else None

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.passed, self.graded)


def summarize(outcomes: Iterable[Outcome]) -> RateSummary:
    counts = Counter(outcomes)
    return RateSummary(
        passed=counts[Outcome.PASS],
        failed=counts[Outcome.FAIL],
        harness_errors=counts[Outcome.HARNESS_ERROR],
    )


class Flip(Model):
    key: str
    base: Outcome
    candidate: Outcome


def flips(base: Mapping[str, Outcome], candidate: Mapping[str, Outcome]) -> list[Flip]:
    """Keys whose graded outcome changed between pass and fail. Harness errors are not flips."""
    graded = {Outcome.PASS, Outcome.FAIL}
    return [
        Flip(key=k, base=base[k], candidate=candidate[k])
        for k in sorted(base.keys() & candidate.keys())
        if base[k] in graded and candidate[k] in graded and base[k] != candidate[k]
    ]


def mcnemar_exact(regressions: int, improvements: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pairs of a paired comparison.

    Pairing old and new versions on the *same* cases removes case-difficulty noise, so far fewer
    cases are needed than comparing two independent pass rates. Under the null hypothesis each
    discordant pair is equally likely to go either way: Binomial(n, 0.5).
    """
    n = regressions + improvements
    if n == 0:
        return 1.0
    k = min(regressions, improvements)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (1 << n)
    return min(1.0, 2 * tail)


def half_width(passed: int, n: int) -> float:
    """Half the width of the 95% Wilson interval: the smallest change one run can resolve."""
    lo, hi = wilson_interval(passed, n)
    return (hi - lo) / 2
