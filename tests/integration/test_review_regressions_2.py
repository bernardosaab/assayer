"""Regression tests for the second review (of the first round of fixes)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest

from assayer.adapters.store.sqlite_store import SqliteStore
from assayer.adapters.system.clock import FixedClock
from assayer.adapters.system.git import FakeVcs
from assayer.application.queries import grading_view
from assayer.application.report import build_timeline
from assayer.application.run_eval import RunEval, RunOptions
from assayer.application.validate import validate_eval
from assayer.domain.errors import ConfigError
from assayer.domain.messages import Message, ToolCall
from assayer.domain.outputs import TerminationReason, ToolEvent, Transcript
from assayer.ports.sut import ToolError
from tests.support.harness import REPO, load_spec, scripted, task_harness
from tests.unit.builtin.test_hr_environment import invoke, make_env

RESUME = str(REPO / "evals" / "resume_extraction")


def _seeded(status: str, start: Any, end: Any) -> dict[str, Any]:
    return {
        "pto_requests": [
            {
                "id": "PTO-0001",
                "employee_id": "E100",
                "start_date": start,
                "end_date": end,
                "days": 2,
                "status": status,
            }
        ]
    }


def test_seeded_unquoted_request_dates_do_not_crash_the_overlap_check() -> None:
    env = make_env(_seeded("pending_approval", dt.date(2026, 12, 1), dt.date(2026, 12, 2)))
    result = invoke(
        env,
        "submit_pto_request",
        employee_id="E100",
        start_date="2026-12-07",
        end_date="2026-12-08",
    )
    assert result["request_id"] == "PTO-0002"
    again = invoke(
        env,
        "submit_pto_request",
        employee_id="E100",
        start_date="2026-12-01",
        end_date="2026-12-02",
    )
    assert again["already_existed"] is True  # the seeded (date-typed) request is recognised


def test_overlapping_an_approved_request_is_rejected() -> None:
    env = make_env(_seeded("approved", "2026-12-01", "2026-12-05"))
    with pytest.raises(ToolError, match="overlapping request: PTO-0001"):
        invoke(
            env,
            "submit_pto_request",
            employee_id="E100",
            start_date="2026-12-03",
            end_date="2026-12-04",
        )


@pytest.mark.parametrize("topic", ["--", "__"])
def test_wordless_topic_matches_nothing(topic: str) -> None:
    with pytest.raises(ToolError, match="no words"):
        invoke(make_env(), "lookup_policy", topic=topic)


def test_validate_reports_runner_type_errors_instead_of_crashing() -> None:
    spec = load_spec("resume_extraction")
    assert spec.task is not None
    for args in ({"system_prompt": "dup"}, {"derive": ["not", "a", "mapping"]}):
        broken = spec.model_copy(update={"task": spec.task.model_copy(update={"args": args})})
        report = validate_eval(broken)
        assert not report.ok, args


def test_derive_must_be_a_mapping() -> None:
    from assayer.application.registry import build_task_runner

    with pytest.raises(ConfigError, match="must be a mapping"):
        build_task_runner("extraction", system_prompt="x", derive=["years"])


async def test_grading_view_uses_the_runs_stored_cases(tmp_path: Path) -> None:
    store = await SqliteStore.open(tmp_path / "r.sqlite", clock=FixedClock())
    spec = load_spec("resume_extraction")
    result = await RunEval(
        store, task_harness(spec, scripted(spec), store), clock=FixedClock(), vcs=FakeVcs()
    )(RunOptions(run_id="r1"))
    renamed = spec.model_copy(
        update={"cases": [c.model_copy(update={"id": f"new-{c.id}"}) for c in spec.cases]}
    )
    view = await grading_view(store, result.meta, renamed)
    await store.close()
    assert view.checks is not None
    assert len(view.checks["jane-doe"]) == 4  # keyed by the run's cases, not today's file


def test_timeline_tolerates_out_of_order_events() -> None:
    a = ToolCall(id="1", name="a", arguments={})
    b = ToolCall(id="2", name="b", arguments={})
    transcript = Transcript(
        messages=(Message.user("go"), Message.assistant(None, (a, b)), Message.assistant("done")),
        tool_events=(ToolEvent(turn=1, call=b, result="B"), ToolEvent(turn=1, call=a, result="A")),
        termination=TerminationReason.AGENT_ENDED,
    )
    tools = [i for i in build_timeline(transcript) if i.actor == "tool"]
    assert [t.result for t in tools] == ["A", "B"]
    assert all(t.error is None for t in tools)


def test_mismatch_note_names_only_the_stale_side() -> None:
    from assayer.application.queries import GradingView, grading_mismatch_note
    from tests.support.builders import a_run

    base, cand = a_run("base"), a_run("cand")
    current = GradingView("g-new", "j-new", {}, current=True)
    stale = GradingView("g-old", "j-old", None, current=False)
    assert grading_mismatch_note(base, current, cand, current) is None
    note = grading_mismatch_note(base, current, cand, stale)
    assert note is not None
    assert "assayer replay cand --into cand-regraded" in note
    assert "replay base" not in note  # the base is current; only the stale side is named


# --- third review ----------------------------------------------------------------------------


async def test_replay_reports_skipped_judgements_to_the_observer(tmp_path: Path) -> None:
    from assayer.application.events import Judged, RecordingObserver
    from assayer.application.replay import ReplayRun
    from assayer.domain.money import Money

    store = await SqliteStore.open(tmp_path / "r.sqlite", clock=FixedClock())
    spec = load_spec("resume_extraction")
    harness = task_harness(spec, scripted(spec), store)
    await RunEval(store, harness, clock=FixedClock(), vcs=FakeVcs())(RunOptions(run_id="base"))
    assert spec.judge is not None
    stricter = spec.model_copy(
        update={"judge": spec.judge.model_copy(update={"instructions": "be strict"})}
    )
    import dataclasses

    observer = RecordingObserver()
    await ReplayRun(
        store, dataclasses.replace(harness, spec=stricter), clock=FixedClock(), observer=observer
    )("base", "re", rejudge=False, budget=Money.zero())
    await store.close()
    skipped = [e for e in observer.of(Judged) if e.skipped]
    assert len(skipped) == 6


def test_timeline_prefers_the_same_turn_for_identical_calls() -> None:
    call = ToolCall(id="c", name="get_pto_balance", arguments={})
    transcript = Transcript(
        messages=(
            Message.user("one"),
            Message.assistant(None, (call,)),
            Message.assistant("first"),
            Message.user("two"),
            Message.assistant(None, (call,)),
            Message.assistant("second"),
        ),
        # Recorded out of order: turn 2's event first.
        tool_events=(
            ToolEvent(turn=2, call=call, result="turn-2"),
            ToolEvent(turn=1, call=call, result="turn-1"),
        ),
        termination=TerminationReason.AGENT_ENDED,
    )
    tools = [(i.turn, i.result) for i in build_timeline(transcript) if i.actor == "tool"]
    assert tools == [(1, "turn-1"), (2, "turn-2")]


async def test_report_shows_raw_output_and_graded_form(tmp_path: Path) -> None:
    from assayer.application.report import build_run_report

    store = await SqliteStore.open(tmp_path / "r.sqlite", clock=FixedClock())
    spec = load_spec("resume_extraction")
    result = await RunEval(
        store, task_harness(spec, scripted(spec), store), clock=FixedClock(), vcs=FakeVcs()
    )(RunOptions(run_id="r1"))
    report = await build_run_report(
        store, result.meta, await grading_view(store, result.meta, spec), generated_by="t"
    )
    await store.close()
    views = {v.attempt.case_id: v for v in report.attempts}
    jane = views["jane-doe"]
    assert '"first_job_start_year": 2016' in jane.output_text  # what the model returned
    assert '"years_experience": 10' in jane.graded_text  # what was graded (derived in code)
    assert "years_experience" not in jane.output_text
    priya = views["priya-sharma"]
    assert priya.output_text.startswith("Sure!")  # unparseable: raw text, nothing derived
    assert priya.graded_text == ""
