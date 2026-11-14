"""Replay separates grading changes from agent changes; reports are stable byte-for-byte."""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from assayer.adapters.store.sqlite_store import SqliteStore
from assayer.adapters.system.clock import FixedClock
from assayer.adapters.system.git import FakeVcs
from assayer.application.compare import compare
from assayer.application.queries import grading_view
from assayer.application.replay import ReplayRun
from assayer.application.report import DiffReport, build_run_report
from assayer.application.run_eval import RunEval, RunOptions
from assayer.application.scoring import grader_version
from assayer.cli.html import JinjaReportRenderer
from assayer.domain.fixtures import CheckSpec
from assayer.domain.money import Money
from tests.support.harness import load_spec, scripted, task_harness

GOLDEN = Path(__file__).resolve().parents[1] / "golden"


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteStore]:
    s = await SqliteStore.open(tmp_path / "r.sqlite", clock=FixedClock())
    yield s
    await s.close()


async def test_replay_with_changed_checks_makes_zero_model_calls(store: SqliteStore) -> None:
    spec = load_spec("resume_extraction")
    provider = scripted(spec)
    harness = task_harness(spec, provider, store)
    await RunEval(store, harness, clock=FixedClock(), vcs=FakeVcs())(RunOptions(run_id="base"))
    calls = len(provider.calls)

    stricter = [
        CheckSpec(use="set_f1", args={"field": "skills", "threshold": 0.95})
        if c.use == "set_f1"
        else c
        for c in spec.checks
    ]
    new_spec = spec.model_copy(update={"checks": stricter})
    new_harness = dataclasses.replace(harness, spec=new_spec)
    result = await ReplayRun(store, new_harness, clock=FixedClock())(
        "base", "replayed", rejudge=False, budget=Money.zero()
    )
    assert len(provider.calls) == calls  # neither the agent nor the judge was called
    assert result.copied == 6
    assert result.scoring.judgements_reused == 6
    assert result.meta.replay_of == "base"

    comparison = compare(
        "base",
        await store.list_grades("base", grader_version(spec, harness.judge_version)),
        "replayed",
        await store.list_grades("replayed", grader_version(new_spec, harness.judge_version)),
    )
    # Same outputs, stricter grader: the only flip is attributable to grading.
    assert [f.key for f in comparison.flips] == ["amara-okafor#0"]
    assert comparison.regressions == 1


async def test_replay_with_changed_rubric_needs_rejudge(store: SqliteStore) -> None:
    spec = load_spec("resume_extraction")
    provider = scripted(spec)
    harness = task_harness(spec, provider, store)
    await RunEval(store, harness, clock=FixedClock(), vcs=FakeVcs())(RunOptions(run_id="base"))
    assert spec.judge is not None
    new_judge = spec.judge.model_copy(update={"instructions": "Be extra strict."})
    new_spec = spec.model_copy(update={"judge": new_judge})
    new_harness = dataclasses.replace(harness, spec=new_spec)

    cached_only = await ReplayRun(store, new_harness, clock=FixedClock())(
        "base", "cache-only", rejudge=False, budget=Money.zero()
    )
    assert cached_only.scoring.judgements_skipped == 6
    assert cached_only.scoring.notes

    calls = len(provider.calls)
    rejudged = await ReplayRun(store, new_harness, clock=FixedClock())(
        "base", "rejudged", rejudge=True, budget=Money.from_usd("1")
    )
    assert rejudged.scoring.judgements_computed == 6
    assert len(provider.calls) > calls


async def _deterministic_run(store: SqliteStore, run_id: str) -> None:
    spec = load_spec("resume_extraction")
    harness = task_harness(spec, scripted(spec), store)
    await RunEval(store, harness, clock=FixedClock(), vcs=FakeVcs())(
        RunOptions(run_id=run_id, concurrency=1)
    )


def _check_golden(name: str, html: str, update: bool) -> None:
    path = GOLDEN / name
    if update or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8", newline="\n")
    assert html == path.read_text(encoding="utf-8"), f"{name} changed; rerun with --update-golden"


async def test_run_report_matches_golden(store: SqliteStore, update_golden: bool) -> None:
    await _deterministic_run(store, "golden")
    spec = load_spec("resume_extraction")
    meta = await store.get_run("golden")
    assert meta is not None
    view = await grading_view(store, meta, spec)
    assert view.current
    report = await build_run_report(store, meta, view, generated_by="assayer (test)")
    html = JinjaReportRenderer().render_run(report)
    assert "carlos-ruiz" in html
    assert "dropped (quote not in evidence)" in html
    _check_golden("report_run.html", html, update_golden)


async def test_diff_report_matches_golden(store: SqliteStore, update_golden: bool) -> None:
    await _deterministic_run(store, "a")
    await _deterministic_run(store, "b")
    spec = load_spec("resume_extraction")
    version = grader_version(spec, spec.judge_version("fake/judge"))
    base, cand = await store.get_run("a"), await store.get_run("b")
    assert base is not None
    assert cand is not None
    comparison = compare(
        "a", await store.list_grades("a", version), "b", await store.list_grades("b", version)
    )
    html = JinjaReportRenderer().render_diff(
        DiffReport(
            base=base,
            candidate=cand,
            comparison=comparison,
            max_drop=0.05,
            regressed=False,
            generated_by="assayer (test)",
        )
    )
    _check_golden("report_diff.html", html, update_golden)
