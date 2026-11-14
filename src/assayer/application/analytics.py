"""Run-level numbers a person acts on: rates with intervals, reliability, cost, latency, causes.

* **Pass rate** excludes harness errors and has a 95% Wilson interval (ADR-0018).
* **pass^k** is the share of cases whose *every* graded repeat passed; that is reliability.
  **pass@k** is the share with at least one pass, i.e. capability.
* **Cost per successful attempt** divides *all* spend for the run (agent, simulator, judge,
  retries and failures) by the number of passing attempts, the honest unit cost.
* **Failure causes** count failing attempts by the check or criterion that failed, the starting
  point for error analysis.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence

from pydantic import Field

from assayer.domain.attempt import Attempt
from assayer.domain.base import Model
from assayer.domain.fixtures import ConversationFixture, TaskFixture
from assayer.domain.grading import Grade
from assayer.domain.money import Money
from assayer.domain.outcome import Outcome
from assayer.domain.stats import RateSummary, summarize
from assayer.ports.store import LedgerEntry, StoredJudgement


class JudgeStats(Model):
    judgements: int = 0
    harness_errors: int = 0
    reasks: int = 0
    findings: int = 0
    dropped_findings: int = 0
    raw_final_disagreements: int = 0  # judge said fail but no finding survived verification

    @property
    def drop_rate(self) -> float | None:
        return self.dropped_findings / self.findings if self.findings else None


class RunAnalytics(Model):
    rates: RateSummary
    cases: int
    pass_hat_k: float | None
    pass_at_k: float | None
    total_cost: Money
    cost_by_purpose: dict[str, Money] = Field(default_factory=dict)
    cost_per_success: Money | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    by_kind: dict[str, RateSummary] = Field(default_factory=dict)
    failure_causes: list[tuple[str, int]] = Field(default_factory=list)
    harness_error_kinds: dict[str, int] = Field(default_factory=dict)
    judge: JudgeStats = Field(default_factory=JudgeStats)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Inclusive-method percentile; ``None`` for no data."""
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    return statistics.quantiles(values, n=100, method="inclusive")[round(q * 100) - 1]


def cause_of(reason: str) -> str:
    """Collapse a grade reason to its cause: ``check field_match: ...`` -> ``check:field_match``."""
    head = reason.split(":", 1)[0].strip()
    parts = head.split(" ", 1)
    return f"{parts[0]}:{parts[1]}" if len(parts) == 2 else head


def analyze(
    graded: Sequence[tuple[Attempt, Grade]],
    *,
    cases: Sequence[TaskFixture | ConversationFixture],
    ledger: Sequence[LedgerEntry],
    judgements: Sequence[StoredJudgement] = (),
) -> RunAnalytics:
    outcomes = [g.outcome for _, g in graded]
    per_case: dict[str, list[Outcome]] = defaultdict(list)
    for attempt, g in graded:
        if g.outcome is not Outcome.HARNESS_ERROR:
            per_case[attempt.case_id].append(g.outcome)
    measured = [o for o in per_case.values() if o]
    pass_hat_k = (
        sum(all(x is Outcome.PASS for x in o) for o in measured) / len(measured)
        if measured
        else None
    )
    pass_at_k = (
        sum(any(x is Outcome.PASS for x in o) for o in measured) / len(measured)
        if measured
        else None
    )

    def effective(e: LedgerEntry) -> int:
        if e.status == "settled":
            return e.actual_micros or 0
        return 0 if e.status == "released" else e.reserved_micros

    total = Money(sum(effective(e) for e in ledger))
    by_purpose: dict[str, Money] = defaultdict(Money.zero)
    for e in ledger:
        by_purpose[e.purpose.value] += Money(effective(e))
    passes = outcomes.count(Outcome.PASS)

    durations = [
        d
        for a, g in graded
        if g.outcome is not Outcome.HARNESS_ERROR and (d := a.duration_ms) is not None
    ]

    kind_of = {
        c.id: c.scenario.kind.value if isinstance(c, ConversationFixture) else "task" for c in cases
    }
    by_kind: dict[str, list[Outcome]] = defaultdict(list)
    for attempt, g in graded:
        by_kind[kind_of.get(attempt.case_id, "unknown")].append(g.outcome)

    causes: Counter[str] = Counter()
    for _, g in graded:
        if g.outcome is Outcome.FAIL:
            causes.update({cause_of(r) for r in g.reasons})
    error_kinds = Counter(
        a.error_kind.value
        for a, g in graded
        if g.outcome is Outcome.HARNESS_ERROR and a.error_kind is not None
    )

    stats = JudgeStats(
        judgements=len(judgements),
        harness_errors=sum(j.verdict.verdict_final is None for j in judgements),
        reasks=sum(j.verdict.reasks for j in judgements),
        findings=sum(len(j.verdict.findings) for j in judgements),
        dropped_findings=sum(len(j.verdict.dropped) for j in judgements),
        raw_final_disagreements=sum(
            j.verdict.verdict_raw == "fail" and j.verdict.verdict_final == "pass"
            for j in judgements
        ),
    )
    return RunAnalytics(
        rates=summarize(outcomes),
        cases=len(cases),
        pass_hat_k=pass_hat_k,
        pass_at_k=pass_at_k,
        total_cost=total,
        cost_by_purpose=dict(by_purpose),
        cost_per_success=Money(total.micros // passes) if passes else None,
        latency_p50_ms=percentile(durations, 0.50),
        latency_p95_ms=percentile(durations, 0.95),
        by_kind={k: summarize(v) for k, v in sorted(by_kind.items())},
        # Ties broken by name: Counter order would follow set iteration (hash-randomized).
        failure_causes=sorted(causes.items(), key=lambda kv: (-kv[1], kv[0])),
        harness_error_kinds=dict(error_kinds),
        judge=stats,
    )
