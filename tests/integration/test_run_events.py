"""The run emits a faithful event stream (what the live view renders)."""

from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from assayer.adapters.store.sqlite_store import SqliteStore
from assayer.adapters.system.clock import FixedClock
from assayer.adapters.system.git import FakeVcs
from assayer.application.events import (
    AttemptFinished,
    AttemptStarted,
    Event,
    Graded,
    Judged,
    RecordingObserver,
    StageStarted,
    TurnEvent,
    safe_notify,
)
from assayer.application.run_eval import RunEval, RunOptions
from assayer.cli.live import LiveObserver, _glyphs
from tests.support.harness import load_spec, scripted, task_harness


async def test_task_run_events(tmp_path: Path) -> None:
    store = await SqliteStore.open(tmp_path / "r.sqlite", clock=FixedClock())
    spec = load_spec("resume_extraction")
    observer = RecordingObserver()
    await RunEval(
        store,
        task_harness(spec, scripted(spec), store),
        clock=FixedClock(),
        vcs=FakeVcs(),
        observer=observer,
    )(RunOptions(run_id="r1"))
    await store.close()

    assert observer.of(StageStarted)[0] == StageStarted("execute", 6)
    assert len(observer.of(AttemptStarted)) == 6
    assert all(e.ok for e in observer.of(AttemptFinished))
    judged = {e.case_id: e for e in observer.of(Judged)}
    assert judged["mei-chen"].reasks == 1
    assert judged["amara-okafor"].dropped == 1
    outcomes = {e.case_id: e.outcome for e in observer.of(Graded)}
    assert outcomes["carlos-ruiz"] == "fail"
    assert outcomes["jane-doe"] == "pass"


class _Broken:
    def notify(self, event: Event) -> None:
        raise RuntimeError("observer bug")


def test_observer_failures_never_reach_the_run() -> None:
    safe_notify(_Broken(), StageStarted("execute", 1))  # does not raise


def test_glyphs_fall_back_to_ascii_on_legacy_consoles() -> None:
    class Legacy:
        encoding = "cp1252"

    assert _glyphs(Legacy())["arrow"] == "->"  # type: ignore[arg-type]
    assert _glyphs(Console(file=io.StringIO()))["arrow"] == "→"


def test_live_view_renders_every_event() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, width=100)
    live = LiveObserver(console, trace=True)
    with live:
        for event in (
            StageStarted("execute", 1),
            AttemptStarted("c", 0),
            TurnEvent("c", 1, "tool", "submit_pto_request({})", error="timeout"),
            TurnEvent("c", 1, "agent", "Done → submitted ✓"),
            AttemptFinished("c", 0, ok=True, duration_ms=12.0),
            StageStarted("judge", 1),
            Judged("c", 0, "fail", reasks=1, dropped=1, cached=False),
            Graded("c", 0, "fail", ("check no_false_success: claimed success",)),
        ):
            live.notify(event)
    out = buffer.getvalue()
    assert "→ submit_pto_request({})  ✗ timeout" in out
    assert "FAIL" in out
    assert "1 re-ask, 1 unverified quote dropped" in out
