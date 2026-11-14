"""Terminal rendering with ``rich``. Presentation only: no decisions are made here."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from assayer.application.analytics import RunAnalytics
from assayer.application.compare import Comparison
from assayer.application.run_eval import RunResult
from assayer.domain.attempt import Attempt
from assayer.domain.calibration import CalibrationStats
from assayer.domain.grading import Grade
from assayer.domain.money import Money
from assayer.domain.run import RunMeta

console = Console(highlight=False)


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def money(value: Money | None) -> str:
    return "n/a" if value is None else f"${value.to_usd():.4f}"


def ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f} ms"


def render_run_result(result: RunResult) -> None:
    meta, ex = result.meta, result.execution
    verb = "Resumed" if result.resumed else "Started"
    console.print(f"[bold]{verb} run[/] {meta.run_id}  ({meta.eval_name}, {meta.models.agent})")
    rec = result.recovery
    if rec.reset or rec.interrupted_after_charge or rec.orphaned_reservations:
        console.print(
            f"  recovered: {rec.reset} reset, {rec.interrupted_after_charge} interrupted after "
            f"spend, {rec.orphaned_reservations} orphaned reservations"
        )
    if result.retried:
        console.print(f"  retrying {result.retried} harness-error attempt(s)")
    console.print(
        f"  executed {ex.executed} attempt(s): {ex.succeeded} ok, "
        f"{ex.harness_errors} harness errors"
    )
    if ex.halted_reason:
        console.print(f"  [yellow]halted:[/] {ex.halted_reason}")
    if result.scoring:
        sc = result.scoring
        console.print(
            f"  scored: {sc.checks_computed} new check results, "
            f"{sc.judgements_computed} judgements "
            f"({sc.judgements_reused} reused, {sc.judge_errors} judge errors)"
        )
        for note in sc.notes:
            console.print(f"  [yellow]note:[/] {note}")
        if sc.halted_reason:
            console.print(f"  [yellow]halted:[/] {sc.halted_reason}")
    console.print(f"  status: [bold]{result.status.value}[/]")


def render_analytics(meta: RunMeta, a: RunAnalytics) -> None:
    lo, hi = a.rates.interval
    table = Table(title=f"{meta.run_id} - {meta.eval_name}", show_header=False, box=None)
    table.add_column(style="dim")
    table.add_column()
    table.add_row(
        "pass rate", f"{pct(a.rates.pass_rate)}  (95% CI {lo:.1%}-{hi:.1%}, n={a.rates.graded})"
    )
    table.add_row(
        "outcomes",
        f"{a.rates.passed} pass / {a.rates.failed} fail / {a.rates.harness_errors} harness error",
    )
    table.add_row(f"pass^k (k={meta.repeats})", pct(a.pass_hat_k))
    table.add_row(f"pass@k (k={meta.repeats})", pct(a.pass_at_k))
    table.add_row("total cost", money(a.total_cost))
    table.add_row("cost per success", money(a.cost_per_success))
    table.add_row("latency p50 / p95", f"{ms(a.latency_p50_ms)} / {ms(a.latency_p95_ms)}")
    if a.judge.judgements:
        table.add_row(
            "judge",
            f"{a.judge.judgements} judgements, {a.judge.reasks} re-asks, "
            f"{a.judge.dropped_findings}/{a.judge.findings} findings dropped (unverified quote)",
        )
    console.print(table)
    if len(a.by_kind) > 1:
        kinds = Table("kind", "pass rate", "n", title="by scenario kind")
        for kind, s in a.by_kind.items():
            kinds.add_row(kind, pct(s.pass_rate), str(s.graded))
        console.print(kinds)
    if a.failure_causes:
        causes = Table("cause", "failing attempts", title="failure causes (error analysis)")
        for cause, count in a.failure_causes:
            causes.add_row(cause, str(count))
        console.print(causes)
    if a.harness_error_kinds:
        console.print(f"harness errors (excluded from rates): {a.harness_error_kinds}")


def render_failures(rows: list[tuple[Attempt, Grade]], limit: int) -> None:
    if not rows:
        console.print("No failures.")
        return
    table = Table("case", "rep", "outcome", "reasons", title="failures")
    for attempt, grade in rows[:limit]:
        reasons = "\n".join(grade.reasons) or (attempt.error_msg or "")
        table.add_row(attempt.case_id, str(attempt.repeat), grade.outcome.value, reasons)
    console.print(table)
    if len(rows) > limit:
        console.print(f"... {len(rows) - limit} more")


def render_comparison(c: Comparison, regressed: bool, max_drop: float) -> None:
    table = Table(title=f"{c.base_run} -> {c.candidate_run} (paired)", show_header=False, box=None)
    table.add_column(style="dim")
    table.add_column()
    for label, s in (("base", c.base), ("candidate", c.candidate)):
        lo, hi = s.interval
        table.add_row(label, f"{pct(s.pass_rate)}  (95% CI {lo:.1%}-{hi:.1%}, n={s.graded})")
    delta = "n/a" if c.paired_delta is None else f"{c.paired_delta:+.1%}"
    table.add_row("paired change", f"{delta} over {c.pairs} pairs")
    table.add_row("discordant", f"{c.improvements} improved, {c.regressions} regressed")
    table.add_row("McNemar exact p", f"{c.p_value:.3f}")
    if c.resolvable_change is not None:
        table.add_row(
            "resolvable", f"changes under +/-{c.resolvable_change:.1%} are noise at this n"
        )
    console.print(table)
    for f in c.flips:
        console.print(f"  {f.key}: {f.base.value} -> {f.candidate.value}")
    verdict = "[red]REGRESSION[/]" if regressed else "[green]within gate[/]"
    console.print(f"gate (max drop {max_drop:.1%}): {verdict}")


def render_calibration(stats: CalibrationStats, observed_pass_rate: float | None) -> None:
    if stats.n == 0:
        console.print("No human labels yet: run `assayer calibrate sample`, label, then import.")
        return
    table = Table(title="judge vs people", show_header=False, box=None)
    table.add_column(style="dim")
    table.add_column()
    table.add_row(
        "labelled items",
        f"{stats.n} ({stats.human_positives} pass / {stats.human_negatives} fail by people)",
    )
    table.add_row("agreement", pct(stats.agreement))
    table.add_row("Cohen's kappa", "n/a" if stats.kappa is None else f"{stats.kappa:.2f}")
    table.add_row("TPR (judge pass | human pass)", pct(stats.tpr))
    table.add_row("TNR (judge fail | human fail)", pct(stats.tnr))
    if observed_pass_rate is not None:
        corrected = stats.corrected_rate(observed_pass_rate)
        table.add_row("judge pass rate (observed)", pct(observed_pass_rate))
        table.add_row("corrected for judge error", pct(corrected))
    console.print(table)
    if stats.n < 30:
        console.print("[yellow]note:[/] fewer than 30 labels; treat these rates as rough.")
