"""CLI smoke tests: every command runs and exit codes carry meaning."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from assayer import __version__
from assayer.cli.app import EXIT_HALTED, app
from tests.support.harness import REPO

RESUME = str(REPO / "evals" / "resume_extraction")


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "assayer.toml"
    prices = (REPO / "prices.yaml").as_posix()
    path.write_text(
        f'store_path = "results.sqlite"\nprices_path = "{prices}"\n[defaults]\nconcurrency = 2\n',
        encoding="utf-8",
    )
    return path


def invoke(config: Path, *args: str) -> tuple[int, str]:
    result = CliRunner().invoke(app, ["--config", str(config), *args])
    return result.exit_code, result.output


def test_version_flag() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_run_then_inspect(config: Path) -> None:
    code, out = invoke(config, "run", RESUME, "--run-id", "cli-1")
    assert code == 0, out
    assert "pass rate" in out
    assert "66.7%" in out

    code, out = invoke(config, "run", RESUME, "--run-id", "cli-1")
    assert code == 0
    assert "Resumed run cli-1" in out
    assert "executed 0 attempt(s)" in out

    for command in (
        ["summary", "cli-1", "--eval", RESUME],
        ["failures", "cli-1", "--eval", RESUME],
        ["budget", "cli-1"],
        ["runs"],
        ["score", "cli-1", "--eval", RESUME],
    ):
        code, out = invoke(config, *command)
        assert code == 0, (command, out)
    _, out = invoke(config, "failures", "cli-1", "--eval", RESUME)
    assert "carlos-ruiz" in out


def test_budget_halt_exit_code(config: Path) -> None:
    code, out = invoke(
        config, "run", RESUME, "--run-id", "cli-2", "--budget-usd", "0.001", "--concurrency", "1"
    )
    assert code == EXIT_HALTED, out
    assert "halted" in out


def test_errors_are_reported_cleanly(config: Path) -> None:
    code, out = invoke(config, "summary", "no-such-run")
    assert code == 2
    assert "unknown run" in out
    code, out = invoke(config, "run", RESUME, "--profile", "nope")
    assert code == 2
    assert "unknown profile" in out


def test_validate(config: Path) -> None:
    code, out = invoke(config, "validate", RESUME)
    assert code == 0
    assert "6 cases, ok" in out


HR = str(REPO / "evals" / "hr_assistant")


def test_report_replay_diff_and_calibration_flow(config: Path, tmp_path: Path) -> None:
    assert invoke(config, "run", RESUME, "--run-id", "base")[0] == 0
    code, out = invoke(config, "report", "base", "--eval", RESUME, "-o", str(tmp_path / "r.html"))
    assert code == 0, out
    assert (tmp_path / "r.html").read_text(encoding="utf-8").startswith("<!doctype html>")

    code, out = invoke(config, "replay", "base", "--into", "again", "--eval", RESUME)
    assert code == 0, out
    assert "6 judgements reused, 0 new" in out

    code, out = invoke(
        config, "diff", "base", "again", "--eval", RESUME, "--html", str(tmp_path / "d.html")
    )
    assert code == 0, out
    assert "within gate" in out

    labels = tmp_path / "labels.yaml"
    code, out = invoke(config, "calibrate", "sample", "base", "-o", str(labels), "--eval", RESUME)
    assert code == 0, out
    code, out = invoke(config, "calibrate", "stats", "base")
    assert "No human labels yet" in out
    labelled = labels.read_text(encoding="utf-8").replace(
        "human_verdict: null", "human_verdict: pass"
    )
    labels.write_text(labelled, encoding="utf-8")
    code, out = invoke(config, "calibrate", "import", str(labels), "--labeller", "ann")
    assert code == 0, out
    code, out = invoke(config, "calibrate", "stats", "base")
    assert code == 0, out
    assert "TPR" in out


def test_min_pass_rate_gate(config: Path) -> None:
    code, out = invoke(config, "run", RESUME, "--run-id", "g", "--min-pass-rate", "0.9")
    assert code == 1, out
    assert "gate failed" in out


def test_conversation_eval_runs_offline(config: Path) -> None:
    code, out = invoke(config, "run", HR, "--run-id", "hr")
    assert code == 0, out
    assert "by scenario kind" in out
    assert "no_false_success" in out


def test_providers_check_for_fake_profile(config: Path) -> None:
    code, out = invoke(config, "providers", "--profile", "fake")
    assert code == 0, out
    assert "scripted, offline" in out
