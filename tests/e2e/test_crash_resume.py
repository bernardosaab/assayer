"""Kill the real CLI mid-call (``os._exit``, like ``kill -9``) and prove the resume semantics.

The process dies on the 3rd model call, *after* the cost guard reserved money for it:

* attempts finished before the crash are kept as they are;
* the in-flight attempt had spend reserved, so on resume it becomes a visible
  ``interrupted_after_charge`` harness error instead of being silently re-spent;
* its orphaned reservation still counts against the budget;
* ``--retry-errors`` then completes the run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from assayer.adapters.providers.faults import CRASH_EXIT_CODE
from assayer.adapters.store.sqlite_store import SqliteStore
from assayer.adapters.system.clock import FixedClock
from assayer.cli.container import CRASH_ENV
from assayer.domain.attempt import AttemptStatus
from assayer.domain.errors import ErrorKind
from tests.support.harness import REPO

RESUME = str(REPO / "evals" / "resume_extraction")


def _cli(config: Path, *args: str, env_extra: dict[str, str] | None = None) -> int:
    env = {k: v for k, v in os.environ.items() if k != CRASH_ENV}
    env.update(env_extra or {})
    done = subprocess.run(
        [sys.executable, "-m", "assayer", "--config", str(config), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return done.returncode


@pytest.mark.slow
async def test_crash_mid_call_then_resume(tmp_path: Path) -> None:
    config = tmp_path / "assayer.toml"
    config.write_text(
        f'store_path = "results.sqlite"\nprices_path = "{(REPO / "prices.yaml").as_posix()}"\n',
        encoding="utf-8",
    )
    args = ("run", RESUME, "--run-id", "crash-1", "--concurrency", "1")

    assert _cli(config, *args, env_extra={CRASH_ENV: "3"}) == CRASH_EXIT_CODE

    store = await SqliteStore.open(tmp_path / "results.sqlite", clock=FixedClock())
    statuses = [a.status for a in await store.list_attempts("crash-1")]
    assert statuses.count(AttemptStatus.SUCCEEDED) == 2
    assert statuses.count(AttemptStatus.IN_PROGRESS) == 1  # the crash left its trace
    [reserved] = [e for e in await store.ledger("crash-1") if e.status == "reserved"]
    await store.close()

    assert _cli(config, *args) == 0  # resume: completes, surfaces the interrupted attempt

    store = await SqliteStore.open(tmp_path / "results.sqlite", clock=FixedClock())
    attempts = await store.list_attempts("crash-1")
    interrupted = [a for a in attempts if a.status is AttemptStatus.HARNESS_ERROR]
    assert len(interrupted) == 1
    assert interrupted[0].error_kind is ErrorKind.INTERRUPTED_AFTER_CHARGE
    statuses_by_entry = {e.entry_id: e.status for e in await store.ledger("crash-1")}
    assert statuses_by_entry[reserved.entry_id] == "orphaned"  # still counted, never forgotten
    await store.close()

    assert _cli(config, *args, "--retry-errors") == 0

    store = await SqliteStore.open(tmp_path / "results.sqlite", clock=FixedClock())
    assert {a.status for a in await store.list_attempts("crash-1")} == {AttemptStatus.SUCCEEDED}
    await store.close()
