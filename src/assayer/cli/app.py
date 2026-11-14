"""Command-line interface. Thin by design: parse arguments, call a use case, render the result."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.table import Table

from assayer import __version__
from assayer.adapters.fixtures.yaml_repository import read_yaml
from assayer.adapters.providers.openai_compat import OpenAICompatProvider
from assayer.application.calibration import (
    blind_item,
    judge_vs_human,
    labels_from_items,
    stratified_sample,
)
from assayer.application.compare import compare
from assayer.application.queries import failures, run_analytics
from assayer.application.replay import ReplayRun
from assayer.application.report import DiffReport, build_run_report
from assayer.application.run_eval import RunEval, RunOptions
from assayer.application.scoring import JUDGE_ID, ScoreRun, grader_version, judge_inputs
from assayer.application.validate import validate_eval
from assayer.cli import render
from assayer.cli.container import Container, build_container, load_eval
from assayer.cli.html import JinjaReportRenderer, write
from assayer.cli.live import LiveObserver
from assayer.config import Settings, load_settings
from assayer.domain.errors import AssayerError
from assayer.domain.money import Money
from assayer.domain.outputs import parse_output
from assayer.domain.run import RunMeta, RunStatus

app = typer.Typer(
    name="assayer",
    help="Durable, budget-guarded evals for LLM tasks and conversational agents.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

EXIT_HALTED = 3
EXIT_REGRESSION = 1


class _State:
    config: Path = Path("assayer.toml")
    verbose: bool = False


state = _State()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"assayer {__version__}")
        raise typer.Exit


@app.callback()
def _root(
    config: Annotated[Path, typer.Option("--config", "-c", help="Settings file (TOML).")] = Path(
        "assayer.toml"
    ),
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
    version: Annotated[  # noqa: ARG001 - handled eagerly by its callback
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Assayer command line."""
    state.config = config
    state.verbose = verbose
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _settings() -> Settings:
    return load_settings(state.config)


def _run[T](coro_fn: Callable[[], Awaitable[T]]) -> T:
    """Run a coroutine, turning expected errors into a clean message and exit code 2."""
    try:
        return asyncio.run(coro_fn())  # type: ignore[arg-type]
    except AssayerError as exc:
        render.console.print(f"[red]error:[/] {exc}")
        raise typer.Exit(2) from None


async def _with_container[T](
    settings: Settings, fn: Callable[[Container], Awaitable[T]], **kwargs: object
) -> T:
    container = await build_container(
        settings,
        warn=lambda m: render.console.print(f"[yellow]warning:[/] {m}"),
        **kwargs,  # type: ignore[arg-type]
    )
    try:
        return await fn(container)
    finally:
        await container.aclose()


async def _meta(container: Container, run_id: str) -> RunMeta:
    meta = await container.store.get_run(run_id)
    if meta is None:
        raise AssayerError(f"unknown run {run_id!r}; see `assayer runs`")
    return meta


async def _grader_version(c: Container, meta: RunMeta, eval_ref: str | None) -> str:
    """Which grades to read for a run.

    Prefer the eval's *current* graders (so ``score`` after a check change shows the new
    grades); fall back to the version the run was graded with when the current graders have not
    been applied to it (e.g. the base side of a replay diff after the rubric changed).
    """
    try:
        spec = load_eval(eval_ref or meta.eval_name)
    except AssayerError:
        return meta.grader_version
    current = grader_version(spec, spec.judge_version(meta.models.judge))
    if current != meta.grader_version and not await c.store.list_grades(meta.run_id, current):
        return meta.grader_version
    return current


# --------------------------------------------------------------------------- run / score


@app.command()
def run(
    eval_ref: Annotated[str, typer.Argument(metavar="EVAL", help="Eval directory or name.")],
    profile: Annotated[str, typer.Option("--profile", "-p", help="Model profile.")] = "fake",
    run_id: Annotated[str | None, typer.Option("--run-id", help="Reuse to resume a run.")] = None,
    repeats: Annotated[int | None, typer.Option(min=1, help="Attempts per case.")] = None,
    concurrency: Annotated[int | None, typer.Option(min=1)] = None,
    budget_usd: Annotated[
        str | None, typer.Option("--budget-usd", help="Hard spend cap for this run.")
    ] = None,
    retry_errors: Annotated[
        bool, typer.Option("--retry-errors", help="Re-run harness-error attempts.")
    ] = False,
    no_score: Annotated[bool, typer.Option("--no-score", help="Execute only.")] = False,
    no_judge: Annotated[bool, typer.Option("--no-judge", help="Skip the LLM judge.")] = False,
    force_new_round: Annotated[
        int, typer.Option("--round", min=1, help="Start a new round under an existing run id.")
    ] = 1,
    allow_unpriced: Annotated[
        bool, typer.Option("--allow-unpriced", help="Treat models missing from prices as free.")
    ] = False,
    live: Annotated[
        bool | None,
        typer.Option("--live/--no-live", help="Live progress (default: on in a terminal)."),
    ] = None,
    trace: Annotated[
        bool, typer.Option("--trace", help="Stream conversations turn by turn, with tool calls.")
    ] = False,
    min_pass_rate: Annotated[
        float | None,
        typer.Option("--min-pass-rate", min=0, max=1, help="Exit 1 if the pass rate is lower."),
    ] = None,
) -> None:
    """Execute and score an eval. Re-running with the same --run-id resumes it."""
    settings = _settings()
    d = settings.defaults
    rid = run_id or f"{Path(eval_ref).name}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    options = RunOptions(
        run_id=rid,
        repeats=repeats or d.repeats,
        budget=Money.from_usd(Decimal(budget_usd) if budget_usd else d.budget_usd),
        concurrency=concurrency or d.concurrency,
        circuit_breaker=d.circuit_breaker,
        retry_errors=retry_errors,
        max_tries=d.max_tries,
        score=not no_score,
        judge=not no_judge,
        round=force_new_round,
    )

    async def go(c: Container) -> tuple[RunStatus, float | None]:
        assert c.harness is not None
        show = render.console.is_terminal if live is None else live
        if show or trace:
            with LiveObserver(render.console, trace=trace) as observer:
                result = await RunEval(
                    c.store, c.harness, clock=c.clock, vcs=c.vcs, observer=observer
                )(options)
        else:
            result = await RunEval(c.store, c.harness, clock=c.clock, vcs=c.vcs)(options)
        render.render_run_result(result)
        pass_rate = None
        if result.scoring is not None:
            version = grader_version(c.harness.spec, c.harness.judge_version)
            analytics = await run_analytics(c.store, result.meta, version)
            render.render_analytics(result.meta, analytics)
            pass_rate = analytics.rates.pass_rate
        render.console.print(f"\nrun id: [bold]{rid}[/]")
        return result.status, pass_rate

    status, pass_rate = _run(
        lambda: _with_container(
            settings, go, eval_ref=eval_ref, profile_name=profile, allow_unpriced=allow_unpriced
        )
    )
    if status is RunStatus.HALTED:
        raise typer.Exit(EXIT_HALTED)
    if min_pass_rate is not None and (pass_rate is None or pass_rate < min_pass_rate):
        render.console.print(f"[red]gate failed:[/] pass rate below {min_pass_rate:.1%}")
        raise typer.Exit(EXIT_REGRESSION)


@app.command()
def score(
    run_id: Annotated[str, typer.Argument(metavar="RUN")],
    eval_ref: Annotated[
        str | None, typer.Option("--eval", help="Defaults to the run's eval.")
    ] = None,
    profile: Annotated[str, typer.Option("--profile", "-p")] = "fake",
    no_judge: Annotated[bool, typer.Option("--no-judge")] = False,
) -> None:
    """(Re)score stored outputs: checks, judge, grades. Only missing work is done."""
    settings = _settings()

    async def go(c: Container) -> None:
        assert c.harness is not None
        meta = await _meta(c, run_id)
        report = await ScoreRun(c.store, c.harness)(meta.run_id, judge=not no_judge)
        render.console.print(
            f"checks: {report.checks_computed} new; judgements: {report.judgements_computed} new, "
            f"{report.judgements_reused} reused; grades: {report.grades}"
        )
        version = grader_version(c.harness.spec, c.harness.judge_version)
        render.render_analytics(meta, await run_analytics(c.store, meta, version))

    async def resolve() -> str:
        if eval_ref:
            return eval_ref
        container = await build_container(settings)
        try:
            return (await _meta(container, run_id)).eval_name
        finally:
            await container.aclose()

    ref = _run(resolve)
    _run(lambda: _with_container(settings, go, eval_ref=ref, profile_name=profile))


# --------------------------------------------------------------------------- inspect


@app.command()
def summary(
    run_id: Annotated[str, typer.Argument(metavar="RUN")],
    eval_ref: Annotated[str | None, typer.Option("--eval")] = None,
) -> None:
    """Rates with intervals, pass^k, cost per success, latency and failure causes."""

    async def go(c: Container) -> None:
        meta = await _meta(c, run_id)
        version = await _grader_version(c, meta, eval_ref)
        render.render_analytics(meta, await run_analytics(c.store, meta, version))

    _run(lambda: _with_container(_settings(), go))


@app.command("failures")
def failures_cmd(
    run_id: Annotated[str, typer.Argument(metavar="RUN")],
    eval_ref: Annotated[str | None, typer.Option("--eval")] = None,
    limit: Annotated[int, typer.Option(min=1)] = 50,
) -> None:
    """List failing and errored attempts with their reasons: the hill-climbing view."""

    async def go(c: Container) -> None:
        meta = await _meta(c, run_id)
        rows = await failures(c.store, meta.run_id, await _grader_version(c, meta, eval_ref))
        render.render_failures(rows, limit)

    _run(lambda: _with_container(_settings(), go))


@app.command()
def runs() -> None:
    """List runs in the results store."""

    async def go(c: Container) -> None:
        table = Table("run id", "eval", "status", "agent model", "repeats", "created", "git")
        for meta, status in await c.store.list_runs():
            table.add_row(
                meta.run_id,
                meta.eval_name,
                status.value,
                meta.models.agent,
                str(meta.repeats),
                f"{meta.created_at:%Y-%m-%d %H:%M}",
                (meta.git_sha or "-")[:8] + ("*" if meta.git_dirty else ""),
            )
        render.console.print(table)

    _run(lambda: _with_container(_settings(), go))


@app.command()
def budget(run_id: Annotated[str, typer.Argument(metavar="RUN")]) -> None:
    """Show a run's cost ledger: reserved vs actual spend per call."""

    async def go(c: Container) -> None:
        meta = await _meta(c, run_id)
        committed = await c.store.committed(run_id)
        render.console.print(
            f"budget {render.money(Money(meta.budget_micros))}, committed {render.money(committed)}"
        )
        table = Table("#", "purpose", "model", "status", "reserved", "actual", "tokens in/out")
        for e in await c.store.ledger(run_id):
            table.add_row(
                str(e.entry_id),
                e.purpose.value,
                e.model,
                e.status,
                render.money(Money(e.reserved_micros)),
                render.money(Money(e.actual_micros)) if e.actual_micros is not None else "-",
                f"{e.prompt_tokens or '-'}/{e.completion_tokens or '-'}",
            )
        render.console.print(table)

    _run(lambda: _with_container(_settings(), go))


@app.command()
def validate(eval_ref: Annotated[str, typer.Argument(metavar="EVAL")]) -> None:
    """Check an eval definition before spending anything."""
    try:
        report = validate_eval(load_eval(eval_ref))
    except AssayerError as exc:
        render.console.print(f"[red]error:[/] {exc}")
        raise typer.Exit(2) from None
    for issue in report.issues:
        color = "red" if issue.severity == "error" else "yellow"
        render.console.print(f"[{color}]{issue.severity}[/] {issue.where}: {issue.message}")
    verdict = "[green]ok[/]" if report.ok else "[red]invalid[/]"
    render.console.print(f"{report.eval_name}: {report.cases} cases, {verdict}")
    if not report.ok:
        raise typer.Exit(1)


# --------------------------------------------------------------------------- report / diff / replay


@app.command()
def report(
    run_id: Annotated[str, typer.Argument(metavar="RUN")],
    eval_ref: Annotated[str | None, typer.Option("--eval")] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="HTML file to write.")] = None,
) -> None:
    """Write a static HTML report for a run."""
    settings = _settings()

    async def go(c: Container) -> Path:
        meta = await _meta(c, run_id)
        version = await _grader_version(c, meta, eval_ref)
        model = await build_run_report(
            c.store, meta, version, generated_by=f"assayer {__version__}"
        )
        return write(
            JinjaReportRenderer().render_run(model), out or settings.reports_dir / f"{run_id}.html"
        )

    path = _run(lambda: _with_container(settings, go))
    render.console.print(f"report: {path}")


@app.command()
def diff(
    base_run: Annotated[str, typer.Argument(metavar="BASE")],
    candidate_run: Annotated[str, typer.Argument(metavar="CANDIDATE")],
    eval_ref: Annotated[str | None, typer.Option("--eval")] = None,
    fail_on_drop: Annotated[
        float, typer.Option(help="Gate: max allowed drop in paired pass rate (0.05 = 5 points).")
    ] = 0.0,
    html: Annotated[Path | None, typer.Option("--html", help="Also write an HTML diff.")] = None,
) -> None:
    """Paired comparison of two runs; exits 1 if the candidate regresses past the gate."""

    async def go(c: Container) -> bool:
        base, cand = await _meta(c, base_run), await _meta(c, candidate_run)
        comparison = compare(
            base_run,
            await c.store.list_grades(base_run, await _grader_version(c, base, eval_ref)),
            candidate_run,
            await c.store.list_grades(candidate_run, await _grader_version(c, cand, eval_ref)),
        )
        regressed = comparison.regressed(fail_on_drop)
        render.render_comparison(comparison, regressed, fail_on_drop)
        if html is not None:
            model = DiffReport(
                base=base,
                candidate=cand,
                comparison=comparison,
                max_drop=fail_on_drop,
                regressed=regressed,
                generated_by=f"assayer {__version__}",
            )
            render.console.print(f"report: {write(JinjaReportRenderer().render_diff(model), html)}")
        return regressed

    if _run(lambda: _with_container(_settings(), go)):
        raise typer.Exit(EXIT_REGRESSION)


@app.command()
def replay(
    run_id: Annotated[str, typer.Argument(metavar="RUN")],
    into: Annotated[str, typer.Option("--into", help="Id for the new replay run.")],
    eval_ref: Annotated[str | None, typer.Option("--eval")] = None,
    profile: Annotated[str, typer.Option("--profile", "-p")] = "fake",
    rejudge: Annotated[
        bool, typer.Option("--rejudge", help="Allow judge calls for evidence not in the cache.")
    ] = False,
    budget_usd: Annotated[str, typer.Option("--budget-usd")] = "0.50",
) -> None:
    """Re-score a run's stored outputs with the current graders; never calls the agent."""
    settings = _settings()

    async def go(c: Container) -> None:
        assert c.harness is not None
        result = await ReplayRun(c.store, c.harness, clock=c.clock)(
            run_id, into, rejudge=rejudge, budget=Money.from_usd(Decimal(budget_usd))
        )
        s = result.scoring
        render.console.print(
            f"replayed {result.copied} attempts into {into}: {s.checks_computed} checks, "
            f"{s.judgements_reused} judgements reused, {s.judgements_computed} new"
        )
        for note in s.notes:
            render.console.print(f"[yellow]note:[/] {note}")
        version = grader_version(c.harness.spec, c.harness.judge_version)
        render.render_analytics(result.meta, await run_analytics(c.store, result.meta, version))

    ref = eval_ref or _run(lambda: _with_container(settings, lambda c: _eval_name(c, run_id)))
    _run(lambda: _with_container(settings, go, eval_ref=ref, profile_name=profile))


async def _eval_name(c: Container, run_id: str) -> str:
    return (await _meta(c, run_id)).eval_name


# --------------------------------------------------------------------------- calibration

calibrate_app = typer.Typer(help="Judge calibration: the judge's judge.", no_args_is_help=True)
app.add_typer(calibrate_app, name="calibrate")


@calibrate_app.command("sample")
def calibrate_sample(
    run_id: Annotated[str, typer.Argument(metavar="RUN")],
    out: Annotated[Path, typer.Option("--out", "-o")],
    n: Annotated[int, typer.Option(min=1)] = 20,
    seed: Annotated[int, typer.Option()] = 0,
    eval_ref: Annotated[str | None, typer.Option("--eval")] = None,
) -> None:
    """Export a blind, verdict-stratified sample of judgements for people to label."""

    async def go(c: Container) -> int:
        meta = await _meta(c, run_id)
        cases = {case.id: case for case in await c.store.load_cases(run_id)}
        attempts = {a.attempt_id: a for a in await c.store.list_attempts(run_id)}
        spec = load_eval(eval_ref or meta.eval_name)
        items = []
        for j in stratified_sample(await c.store.list_judgements(run_id), n, seed=seed):
            attempt = attempts[j.attempt_id]
            case = cases[attempt.case_id]
            assert attempt.output is not None
            evidence, _ = judge_inputs(case, parse_output(attempt.output))
            criteria = [f"{k.id}: {k.text}" for k in spec.criteria_for(case)]
            items.append(blind_item(j, attempt.case_id, evidence, criteria))
        out.write_text(yaml.safe_dump(items, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return len(items)

    count = _run(lambda: _with_container(_settings(), go))
    render.console.print(f"wrote {count} items to {out}; fill in human_verdict (pass|fail)")


@calibrate_app.command("import")
def calibrate_import(
    labels: Annotated[Path, typer.Argument(help="The labelled YAML file.")],
    labeller: Annotated[str, typer.Option("--labeller", help="Who labelled it.")],
) -> None:
    """Store human verdicts from a labelled sample."""

    async def go(c: Container) -> int:
        parsed = labels_from_items(read_yaml(labels) or [], labeller)
        for label in parsed:
            await c.store.save_calibration(label)
        return len(parsed)

    count = _run(lambda: _with_container(_settings(), go))
    render.console.print(f"imported {count} labels from {labeller}")


@calibrate_app.command("stats")
def calibrate_stats(run_id: Annotated[str, typer.Argument(metavar="RUN")]) -> None:
    """Judge vs people: agreement, kappa, TPR/TNR, and the bias-corrected pass rate."""

    async def go(c: Container) -> None:
        meta = await _meta(c, run_id)
        if meta.judge_version is None:
            raise AssayerError(f"run {run_id!r} has no judge")
        judgements = await c.store.list_judgements(run_id)
        labels = await c.store.list_calibration(JUDGE_ID, meta.judge_version)
        stats = judge_vs_human(judgements, labels)
        finished = [j for j in judgements if j.verdict.verdict_final is not None]
        observed = (
            sum(j.verdict.verdict_final == "pass" for j in finished) / len(finished)
            if finished
            else None
        )
        render.render_calibration(stats, observed)

    _run(lambda: _with_container(_settings(), go))


# --------------------------------------------------------------------------- providers


@app.command()
def providers(
    profile: Annotated[str, typer.Option("--profile", "-p")] = "ollama",
) -> None:
    """Check that a profile's endpoints answer and list the models they serve."""

    async def go() -> bool:
        settings = _settings()
        prof = settings.profile(profile)
        ok = True
        for role_name, role in (
            ("agent", prof.agent),
            ("simulator", prof.simulator),
            ("judge", prof.judge),
        ):
            if role is None:
                continue
            if role.provider == "fake":
                render.console.print(f"{role_name}: {role.model} (scripted, offline)")
                continue
            assert role.base_url is not None
            provider = OpenAICompatProvider(
                name=role.model.split("/", 1)[0], base_url=role.base_url
            )
            try:
                served = await provider.list_models()
            except AssayerError as exc:
                render.console.print(f"[red]{role_name}: {role.base_url} unreachable:[/] {exc}")
                ok = False
                continue
            finally:
                await provider.aclose()
            wanted = role.model.split("/", 1)[-1]
            mark = "[green]ok[/]" if wanted in served else "[red]model not served[/]"
            ok = ok and wanted in served
            render.console.print(f"{role_name}: {role.model} at {role.base_url}: {mark}")
        return ok

    if not _run(go):
        raise typer.Exit(1)


def main() -> None:
    app()
