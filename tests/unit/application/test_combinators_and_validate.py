from __future__ import annotations

from pathlib import Path
from typing import Any

from assayer.application.registry import build_check
from assayer.application.validate import validate_eval
from assayer.builtin import load_builtins
from assayer.domain.evalspec import EvalKind, EvalSpec, TaskSpec
from assayer.domain.fixtures import CheckSpec
from assayer.domain.outputs import TaskOutput
from tests.support.builders import a_task

load_builtins()

CASE = a_task()
GOOD = TaskOutput(raw="{}", parsed={"name": "Jane Doe", "current_employer": "Acme Corp"})
BAD = TaskOutput(raw="{}", parsed={"name": "John", "current_employer": "Acme Corp"})


def _spec(spec: dict[str, Any]) -> CheckSpec:
    return CheckSpec.model_validate(spec)


def test_all_of_any_of_not() -> None:
    name = {"use": "field_match", "args": {"fields": ["name"]}}
    employer = {"use": "field_match", "args": {"fields": ["current_employer"]}}
    all_of = build_check(_spec({"use": "all_of", "args": {"checks": [name, employer]}}))
    any_of = build_check(_spec({"use": "any_of", "args": {"checks": [name, employer]}}))
    negated = build_check(_spec({"use": "not", "args": {"checks": [name]}}))
    assert all_of.evaluate(CASE, GOOD).passed
    assert not all_of.evaluate(CASE, BAD).passed
    assert any_of.evaluate(CASE, BAD).passed
    assert not any_of.evaluate(CASE, TaskOutput(raw="", parsed={})).passed
    assert negated.evaluate(CASE, BAD).passed
    assert not negated.evaluate(CASE, GOOD).passed


def test_check_version_changes_with_args() -> None:
    a = build_check(_spec({"use": "set_f1", "args": {"field": "x", "threshold": 0.8}}))
    b = build_check(_spec({"use": "set_f1", "args": {"field": "x", "threshold": 0.9}}))
    assert a.version != b.version


def test_validate_reports_every_problem() -> None:
    bad_gold = a_task(
        "c1",
        gold={
            "name": {"votes": {"a": "X", "b": "Y"}},
            "employer": {"resolved": "Acme", "traps": ["Acme"]},
            "title": {"votes": {"a": "Dev"}},
        },
        checks=[{"use": "nope"}, {"use": "trap_absent"}, {"use": "trap_absent"}],
    )
    spec = EvalSpec(
        name="broken",
        kind=EvalKind.CONVERSATION,
        root=Path(),
        task=TaskSpec(system_prompt="x"),
        cases=[bad_gold],
    )
    report = validate_eval(spec)
    messages = " | ".join(f"{i.severity}:{i.message}" for i in report.issues)
    assert not report.ok
    assert "needs 'agent' and 'simulator'" in messages
    assert "unknown check 'nope'" in messages
    assert "duplicate check ids" in messages
    assert "unresolved" in messages
    assert "trap equals the expected value" in messages
    assert "warning:one labeller vote" in messages
    assert "cannot run offline" in messages
