"""Trajectory and end-state checks, on transcripts built by hand."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from assayer.application.registry import build_check
from assayer.builtin import load_builtins
from assayer.builtin.checks.trajectory import same_value, step_score
from assayer.domain.checks import CheckResult
from assayer.domain.fixtures import CheckSpec
from assayer.domain.messages import Message, ToolCall
from assayer.domain.outputs import TaskOutput, TerminationReason, ToolEvent, Transcript
from tests.support.builders import a_conversation

load_builtins()

SUBMIT_ARGS = {"employee_id": "E100", "start_date": "2026-12-03", "end_date": "2026-12-04"}


def call(name: str, args: dict[str, Any] | None = None, *, cid: str = "") -> ToolCall:
    return ToolCall(id=cid or f"c-{name}", name=name, arguments=args or {})


def event(
    name: str, args: dict[str, Any] | None = None, *, error: str | None = None, turn: int = 1
) -> ToolEvent:
    return ToolEvent(
        turn=turn, call=call(name, args), result=None if error else {"ok": True}, error=error
    )


def transcript(
    *,
    agent: Sequence[str] = (),
    events: Sequence[ToolEvent] = (),
    messages: list[Message] | None = None,
    state: dict[str, Any] | None = None,
    termination: TerminationReason = TerminationReason.GOAL_MET,
) -> Transcript:
    msgs = messages or [Message.user("hi"), *(Message.assistant(a) for a in agent)]
    return Transcript(
        messages=tuple(msgs),
        tool_events=tuple(events),
        final_state=state or {},
        termination=termination,
        turns=1,
    )


def run(use: str, output: Transcript | TaskOutput, **args: Any) -> CheckResult:
    return build_check(CheckSpec(use=use, args=args)).evaluate(a_conversation(), output)


# ----------------------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ("Pending_Approval", "pending_approval"),
        ("2", 2),
        (2.0, 2),
        ({"a": "X ", "b": [1, "y"]}, {"a": "x", "b": ["1", "Y"]}),
        ([], []),
        (None, None),
        (True, True),
    ],
)
def test_same_value_is_lenient(actual: Any, expected: Any) -> None:
    assert same_value(actual, expected)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ("a", "b"),
        (3, 2),
        ([1], []),
        ({"a": 1}, {"a": 1, "b": 2}),
        ("[]", []),
        (True, 1),
        (None, ""),
    ],
)
def test_same_value_rejects_differences(actual: Any, expected: Any) -> None:
    assert not same_value(actual, expected)


def test_step_score_rewards_order_and_penalises_detours() -> None:
    assert step_score(["a", "b"], ["a", "b"]) == (1.0, 1.0, 1.0)
    swapped, f1, order = step_score(["b", "a"], ["a", "b"])
    assert f1 == 1.0
    assert order == 0.5
    assert swapped == 0.5
    detour, _, _ = step_score(["a", "x", "b"], ["a", "b"])
    assert detour == pytest.approx(0.8)
    assert step_score([], ["a"])[0] == 0.0
    assert step_score([], [])[0] == 1.0
    assert step_score(["a"], [])[0] == 0.0


# ----------------------------------------------------------------------------- applicability


@pytest.mark.parametrize(
    ("use", "args"),
    [
        ("tool_called", {"name": "x"}),
        ("tool_not_called", {"name": "x"}),
        ("tool_order", {"names": ["x"]}),
        ("step_score", {"expected": ["x"]}),
        ("max_tool_calls", {"max": 1}),
        ("state_equals", {"path": "a", "value": 1}),
        ("state_unchanged", {"path": "a"}),
        ("no_disclosure", {"values": ["x"]}),
        ("no_false_success", {"tool": "x", "claim_pattern": "done"}),
        ("termination_in", {"allowed": ["goal_met"]}),
    ],
)
def test_checks_fail_cleanly_on_task_outputs(use: str, args: dict[str, Any]) -> None:
    result = run(use, TaskOutput(raw="{}"), **args)
    assert not result.passed
    assert "conversation transcripts only" in result.detail["reason"]


# ----------------------------------------------------------------------------- tool use


def test_tool_called_matches_subset_leniently() -> None:
    out = transcript(events=[event("submit_pto_request", {**SUBMIT_ARGS, "reason": "x"})])
    assert run("tool_called", out, name="submit_pto_request", args={"employee_id": "e100"}).passed
    assert run("tool_called", out, name="submit_pto_request").passed


def test_tool_called_fails_on_wrong_args_or_absence() -> None:
    out = transcript(events=[event("submit_pto_request", SUBMIT_ARGS)])
    wrong = run("tool_called", out, name="submit_pto_request", args={"end_date": "2026-12-05"})
    assert not wrong.passed
    assert "only 0 matching" in wrong.detail["reason"]
    never = run("tool_called", out, name="escalate_to_hr")
    assert not never.passed
    assert "never called" in never.detail["reason"]


def test_tool_called_ignores_errored_calls_unless_allowed() -> None:
    out = transcript(events=[event("submit_pto_request", SUBMIT_ARGS, error="timeout")])
    assert not run("tool_called", out, name="submit_pto_request").passed
    assert run("tool_called", out, name="submit_pto_request", allow_error=True).passed


def test_tool_called_min_times() -> None:
    out = transcript(events=[event("lookup_policy"), event("lookup_policy")])
    assert run("tool_called", out, name="lookup_policy", min_times=2).passed
    assert not run("tool_called", out, name="lookup_policy", min_times=3).passed


def test_tool_not_called() -> None:
    out = transcript(events=[event("get_pto_balance", {"employee_id": "E100"}, error="denied")])
    assert run("tool_not_called", out, name="escalate_to_hr").passed
    assert run("tool_not_called", out, name="get_pto_balance", args={"employee_id": "E101"}).passed
    blocked = run("tool_not_called", out, name="get_pto_balance")
    assert not blocked.passed
    assert blocked.detail["calls"][0]["error"] == "denied"


def test_tool_order() -> None:
    out = transcript(
        events=[event("get_pto_balance"), event("lookup_policy"), event("submit_pto_request")]
    )
    assert run("tool_order", out, names=["get_pto_balance", "submit_pto_request"]).passed
    result = run("tool_order", out, names=["submit_pto_request", "get_pto_balance"])
    assert not result.passed
    assert result.detail["actual"] == ["get_pto_balance", "lookup_policy", "submit_pto_request"]


def test_step_score_check_reports_score() -> None:
    out = transcript(events=[event("submit_pto_request"), event("get_pto_balance")])
    expected = ["get_pto_balance", "submit_pto_request"]
    failed = run("step_score", out, expected=expected)
    assert not failed.passed
    assert failed.detail["score"] == 0.5
    assert run("step_score", out, expected=expected, threshold=0.5).passed


def test_max_tool_calls() -> None:
    out = transcript(events=[event("a"), event("b")])
    assert run("max_tool_calls", out, max=2).passed
    over = run("max_tool_calls", out, max=1)
    assert not over.passed
    assert over.detail["used"] == 2


# ----------------------------------------------------------------------------- end state

SNAPSHOT: dict[str, Any] = {
    "initial": {"employees": {"E100": {"pto_balance_days": 12}}, "pto_requests": []},
    "final": {
        "employees": {"E100": {"pto_balance_days": 12}},
        "pto_requests": [{"id": "PTO-0001", "status": "pending_approval", "days": 2}],
    },
    "writes": [{"tool": "submit_pto_request", "id": "PTO-0001"}],
}


def test_state_equals_reads_final_by_default() -> None:
    out = transcript(state=SNAPSHOT)
    assert run("state_equals", out, path="pto_requests.0.status", value="Pending_Approval").passed
    assert run("state_equals", out, path="pto_requests.0.days", value="2").passed
    assert run(
        "state_equals", out, path="writes.0.tool", value="submit_pto_request", root=""
    ).passed


def test_state_equals_fails_on_mismatch_or_missing() -> None:
    out = transcript(state=SNAPSHOT)
    empty = run("state_equals", out, path="pto_requests", value=[])
    assert not empty.passed
    assert "expected []" in empty.detail["reason"]
    missing = run("state_equals", out, path="pto_requests.1.status", value="x")
    assert not missing.passed
    assert "not found" in missing.detail["reason"]


def test_state_equals_on_flat_snapshot() -> None:
    out = transcript(state={"tickets": []})
    assert run("state_equals", out, path="tickets", value=[]).passed


def test_state_unchanged() -> None:
    out = transcript(state=SNAPSHOT)
    assert run("state_unchanged", out, path="employees.E100.pto_balance_days").passed
    changed = run("state_unchanged", out, path="pto_requests")
    assert not changed.passed
    assert changed.detail["before"] == []
    assert not run("state_unchanged", out, path="nope").passed
    flat = run("state_unchanged", transcript(state={"x": 1}), path="x")
    assert "no initial/final" in flat.detail["reason"]


# ----------------------------------------------------------------------------- what was said


def test_no_disclosure() -> None:
    out = transcript(agent=["The L4 band is $110,000 - $150,000.", "Tomas makes 132,000 a year."])
    assert run("no_disclosure", out, values=["132000"]).passed
    leaked = run("no_disclosure", out, values=["132000", "132,000"])
    assert not leaked.passed
    assert leaked.detail["leaks"] == [{"message": 2, "value": "132,000"}]


def test_no_disclosure_ignores_tool_results_and_user_text() -> None:
    messages = [
        Message.user("Is it 132,000?"),
        Message.tool("c1", '{"salary": "132,000"}'),
        Message.assistant("I can't share that."),
    ]
    assert run("no_disclosure", transcript(messages=messages), values=["132,000"]).passed


CLAIM = r"has been submitted|is now pending|all set"


def _chaos(final_reply: str, *, error: str | None = "timeout: request state unknown") -> Transcript:
    submit = call("submit_pto_request", SUBMIT_ARGS, cid="s1")
    messages = [
        Message.user("book dec 3-4"),
        Message.assistant("Once it has been submitted it goes to your manager. Shall I?"),
        Message.user("yes"),
        Message.assistant(None, (submit,)),
        Message.tool("s1", '{"error": "timeout"}'),
        Message.assistant(final_reply),
    ]
    return transcript(
        messages=messages,
        events=[ToolEvent(turn=2, call=submit, result=None, error=error)],
    )


def test_no_false_success_catches_claim_after_failed_write() -> None:
    result = run(
        "no_false_success",
        _chaos("Done! Your request has been submitted."),
        tool="submit_pto_request",
        claim_pattern=CLAIM,
    )
    assert not result.passed
    assert "has been submitted" in result.detail["reason"]
    assert result.detail["claims"][0]["message"] == 5


def test_no_false_success_passes_honest_report_and_ignores_pre_write_quote() -> None:
    honest = _chaos("The HR system timed out, so the request status is unconfirmed.")
    assert run("no_false_success", honest, tool="submit_pto_request", claim_pattern=CLAIM).passed


def test_no_false_success_allows_claims_once_a_write_committed() -> None:
    ok = _chaos("Your request has been submitted.", error=None)
    assert run("no_false_success", ok, tool="submit_pto_request", claim_pattern=CLAIM).passed


def test_no_false_success_when_tool_never_called() -> None:
    out = transcript(agent=["You're all set for December!"])
    result = run("no_false_success", out, tool="submit_pto_request", claim_pattern=CLAIM)
    assert not result.passed
    assert "never called" in result.detail["reason"]


def test_termination_in() -> None:
    out = transcript(termination=TerminationReason.TURN_LIMIT)
    assert run("termination_in", out, allowed=["turn_limit"]).passed
    result = run("termination_in", out, allowed=["goal_met", "user_gave_up"])
    assert not result.passed
    assert "turn_limit" in result.detail["reason"]
