"""Read-side queries shared by the CLI and the report renderer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from assayer.application.analytics import RunAnalytics, analyze
from assayer.application.scoring import checks_for, grader_version
from assayer.domain.attempt import Attempt
from assayer.domain.evalspec import EvalSpec
from assayer.domain.grading import Grade
from assayer.domain.outcome import Outcome
from assayer.domain.run import RunMeta
from assayer.ports.store import AttemptRepository, CostLedger, RunRepository, ScoreRepository


class ReadStore(RunRepository, AttemptRepository, ScoreRepository, CostLedger, Protocol):
    pass


@dataclass(frozen=True)
class GradingView:
    """Which stored scores to read for a run, kept mutually consistent.

    ``checks`` maps case id to the (check id, version) pairs the grades used; ``None`` means the
    run is shown as originally graded and every stored check result is listed.
    """

    grader_version: str
    judge_version: str | None
    checks: dict[str, set[tuple[str, str]]] | None
    current: bool


async def grading_view(
    store: ReadStore, meta: RunMeta, spec: EvalSpec | None, *, with_checks: bool = True
) -> GradingView:
    """Prefer the eval's current graders; fall back to the run's own when they were never applied.

    ``score`` or ``replay`` after a grader change produces grades under the current version. A
    run nobody re-scored only has its original grades, which are then shown as such, and
    ``current`` is False so callers can say so instead of mixing versions silently.
    """
    original = GradingView(meta.grader_version, meta.judge_version, None, current=False)
    if spec is None:
        return original
    judge_version = spec.judge_version(meta.models.judge)
    version = grader_version(spec, judge_version)
    if version != meta.grader_version and not await store.list_grades(meta.run_id, version):
        return original
    if not with_checks:  # callers that only need the version skip loading and building checks
        return GradingView(version, judge_version, None, current=True)
    # The run's own (stored) cases, not today's eval file: cases may have been renamed or edited
    # since, and scoring graded the stored ones.
    stored = await store.load_cases(meta.run_id)
    checks = {c.id: {(k.check_id, k.version) for k in checks_for(spec, c)} for c in stored}
    return GradingView(version, judge_version, checks, current=True)


async def run_analytics(store: ReadStore, meta: RunMeta, grader_version: str) -> RunAnalytics:
    return analyze(
        await store.list_grades(meta.run_id, grader_version),
        cases=await store.load_cases(meta.run_id),
        ledger=await store.ledger(meta.run_id),
        judgements=await store.list_judgements(meta.run_id),
    )


async def failures(
    store: ReadStore, run_id: str, grader_version: str
) -> list[tuple[Attempt, Grade]]:
    return [
        (a, g)
        for a, g in await store.list_grades(run_id, grader_version)
        if g.outcome is not Outcome.PASS
    ]


def grading_mismatch_note(
    base: RunMeta, base_view: GradingView, cand: RunMeta, cand_view: GradingView
) -> str | None:
    """Explain a comparison across different graders, naming the side(s) that are stale."""
    if base_view.grader_version == cand_view.grader_version:
        return None
    stale = [m for m, v in ((base, base_view), (cand, cand_view)) if not v.current]
    fix = "; ".join(
        f"assayer replay {m.run_id} --into {m.run_id}-regraded --rejudge "
        f"--profile <profile whose judge is {m.models.judge or 'none'}>, then diff "
        f"against {m.run_id}-regraded"
        for m in stale
    )
    return (
        f"the runs were graded with different graders ({base_view.grader_version} vs "
        f"{cand_view.grader_version}), so differences may come from grading, not the agent."
        + (f" To compare like with like: {fix}." if fix else "")
    )
