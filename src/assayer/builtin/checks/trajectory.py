"""Trajectory and end-state checks for conversation evals.

These are the deterministic sensors of a conversation eval. They run before (and independently
of) the LLM judge and grade what the agent *did*, not what it *said*:

* **Tool selection and arguments**: ``tool_called``, ``tool_not_called``, ``tool_order``,
  ``step_score``.
* **Efficiency**: ``max_tool_calls`` (steps against a budget).
* **End state** ("score the write, not the claim"): ``state_equals``, ``state_unchanged``
  read the environment snapshot stored in ``Transcript.final_state``.
* **Leaks and false claims**: ``no_disclosure`` (a deterministic leak sensor over the agent's
  messages) and ``no_false_success`` (the agent claims a write that never committed).
* **How it ended**: ``termination_in``.

Snapshot convention
-------------------
An environment's ``snapshot()`` is expected to look like::

    {"initial": {...records at the start...},
     "final":   {...records at the end...},
     "writes":  [{"tool": ..., ...}, ...]}       # committed writes only, in order

``state_equals`` resolves its ``path`` under ``final`` by default (``root: final``); set
``root: ""`` to address the whole snapshot (e.g. ``path: writes``). An environment whose
snapshot has no ``final`` key is addressed as-is. ``state_unchanged`` compares the same path
under ``initial`` and ``final``.

Values are compared leniently: strings case-, whitespace- and trailing-dot-insensitively,
numbers numerically (``"2"`` equals ``2.0``), dicts key by key and lists element by element.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from assayer.application.registry import CHECKS
from assayer.builtin.checks.base import Case, Output, SpecCheck, lookup, norm_text
from assayer.domain.checks import CheckResult
from assayer.domain.outputs import ToolEvent, Transcript

# ----------------------------------------------------------------------------- helpers


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def same_value(actual: Any, expected: Any) -> bool:
    """Lenient structural equality (see the module docstring)."""
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(same_value(actual[k], v) for k, v in expected.items())
        )
    if isinstance(expected, list | tuple):
        return (
            isinstance(actual, list | tuple)
            and len(actual) == len(expected)
            and all(same_value(a, e) for a, e in zip(actual, expected, strict=True))
        )
    if isinstance(expected, bool) or expected is None or isinstance(actual, bool) or actual is None:
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, int | float) or isinstance(actual, int | float):
        a, e = _as_number(actual), _as_number(expected)
        return a is not None and e is not None and a == e
    if isinstance(actual, dict | list | tuple):
        return False
    return norm_text(actual) == norm_text(expected)


def contains_subset(arguments: dict[str, Any], subset: dict[str, Any]) -> bool:
    """Every key of ``subset`` is in ``arguments`` with a leniently equal value."""
    return all(k in arguments and same_value(arguments[k], v) for k, v in subset.items())


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence."""
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def step_score(actual: Sequence[str], expected: Sequence[str]) -> tuple[float, float, float]:
    """Order-aware tool-sequence score: ``(score, f1, order)``.

    ``f1`` is the F1 of the multiset overlap of tool names, ``order`` is
    ``LCS(actual, expected) / len(expected)``, and ``score = f1 * order``. An empty expectation
    scores 1.0 only against an empty trajectory.
    """
    if not expected:
        return (1.0, 1.0, 1.0) if not actual else (0.0, 0.0, 0.0)
    overlap = sum((Counter(actual) & Counter(expected)).values())
    precision = overlap / len(actual) if actual else 0.0
    recall = overlap / len(expected)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    order = _lcs(actual, expected) / len(expected)
    return f1 * order, f1, order


def _describe(event: ToolEvent) -> dict[str, Any]:
    return {
        "turn": event.turn,
        "name": event.call.name,
        "arguments": event.call.arguments,
        "error": event.error,
    }


class _TranscriptCheck(SpecCheck):
    """A check that only applies to conversation transcripts."""

    def evaluate(self, case: Case, output: Output) -> CheckResult:  # noqa: ARG002
        if not isinstance(output, Transcript):
            return self.fail(f"{self.spec.use} applies to conversation transcripts only")
        return self.check(output)

    def check(self, transcript: Transcript) -> CheckResult:
        raise NotImplementedError


def _agent_messages(transcript: Transcript) -> list[tuple[int, str]]:
    """``(index, content)`` of every agent message with text, in transcript order."""
    return [
        (i, m.content)
        for i, m in enumerate(transcript.messages)
        if m.role == "assistant" and m.content
    ]


# ----------------------------------------------------------------------------- tool use


@CHECKS.register("tool_called")
class ToolCalled(_TranscriptCheck):
    """The agent called a tool, with the right arguments, and the call succeeded.

    args:
      ``name``: tool name.
      ``args``: optional subset of arguments that must match (lenient comparison).
      ``min_times``: minimum number of matching calls (default 1).
      ``allow_error``: count calls that returned an error (default false: a call the system
      rejected did not *do* anything, so it does not count as the action being taken).
    """

    def check(self, transcript: Transcript) -> CheckResult:
        name: str = self.args["name"]
        subset: dict[str, Any] = self.args.get("args") or {}
        min_times = int(self.args.get("min_times", 1))
        allow_error = bool(self.args.get("allow_error", False))
        same_name = [e for e in transcript.tool_events if e.call.name == name]
        matching = [
            e
            for e in same_name
            if contains_subset(e.call.arguments, subset) and (allow_error or e.error is None)
        ]
        if len(matching) >= min_times:
            return self.ok(matches=len(matching))
        if not same_name:
            return self.fail(f"{name} was never called")
        return self.fail(
            f"{name} called {len(same_name)}x but only {len(matching)} matching "
            f"(args {subset or 'any'}, {'errors allowed' if allow_error else 'no errors'}); "
            f"needed {min_times}",
            calls=[_describe(e) for e in same_name],
        )


@CHECKS.register("tool_not_called")
class ToolNotCalled(_TranscriptCheck):
    """A forbidden action was never even attempted (errored attempts count as attempts).

    args: ``name``; optional ``args`` subset to forbid only calls with those arguments
    (e.g. reading another employee's record).
    """

    def check(self, transcript: Transcript) -> CheckResult:
        name: str = self.args["name"]
        subset: dict[str, Any] = self.args.get("args") or {}
        hits = [
            e
            for e in transcript.tool_events
            if e.call.name == name and contains_subset(e.call.arguments, subset)
        ]
        if hits:
            return self.fail(
                f"{name} was attempted {len(hits)}x but is forbidden here",
                calls=[_describe(e) for e in hits],
            )
        return self.ok()


@CHECKS.register("tool_order")
class ToolOrder(_TranscriptCheck):
    """Tool names occur as an ordered subsequence (e.g. check the balance before submitting).

    args: ``names: [...]``. Other calls may be interleaved; every attempted call counts.
    """

    def check(self, transcript: Transcript) -> CheckResult:
        names: list[str] = list(self.args["names"])
        actual = [e.call.name for e in transcript.tool_events]
        remaining = iter(actual)
        missing = [n for n in names if not any(n == a for a in remaining)]
        if missing:
            return self.fail(
                f"expected {' -> '.join(names)} in order; got {' -> '.join(actual) or 'no calls'}",
                actual=actual,
            )
        return self.ok(actual=actual)


@CHECKS.register("step_score")
class StepScore(_TranscriptCheck):
    """Order-aware F1 of the tool trajectory against a reference, at or above a threshold.

    args: ``expected: [names...]``, ``threshold`` (default 0.8). The score is the F1 of the
    multiset overlap of tool names times an order factor ``LCS / len(expected)``, so a
    trajectory with the right tools in the wrong order, or with extra detours, scores lower.
    """

    def check(self, transcript: Transcript) -> CheckResult:
        expected: list[str] = list(self.args["expected"])
        threshold = float(self.args.get("threshold", 0.8))
        actual = [e.call.name for e in transcript.tool_events]
        score, f1, order = step_score(actual, expected)
        detail = {
            "score": round(score, 4),
            "f1": round(f1, 4),
            "order": round(order, 4),
            "actual": actual,
        }
        if score >= threshold:
            return self.ok(**detail)
        return self.fail(f"step score {score:.2f} < {threshold:.2f}", **detail)


@CHECKS.register("max_tool_calls")
class MaxToolCalls(_TranscriptCheck):
    """Efficiency budget: at most ``max`` tool calls across the whole conversation.

    args: ``max``.
    """

    def check(self, transcript: Transcript) -> CheckResult:
        limit = int(self.args["max"])
        used = len(transcript.tool_events)
        if used > limit:
            return self.fail(f"{used} tool calls > budget {limit}", used=used, budget=limit)
        return self.ok(used=used, budget=limit)


# ----------------------------------------------------------------------------- end state


def _root(snapshot: dict[str, Any], root: str) -> Any:
    if root and root in snapshot:
        return snapshot[root]
    return snapshot


@CHECKS.register("state_equals")
class StateEquals(_TranscriptCheck):
    """The system of record ends in the expected state: score the write, not the claim.

    args: ``path`` (dotted, e.g. ``pto_requests.0.status``), ``value`` (lenient comparison;
    ``[]`` asserts an empty collection), ``root`` (default ``final``; ``""`` for the whole
    snapshot).
    """

    def check(self, transcript: Transcript) -> CheckResult:
        path: str = self.args["path"]
        expected = self.args.get("value")
        base = _root(transcript.final_state, str(self.args.get("root", "final")))
        found, actual = lookup(base, path)
        if not found:
            return self.fail(f"{path} not found in the end state", expected=expected)
        if not same_value(actual, expected):
            return self.fail(
                f"end state {path} is {actual!r}, expected {expected!r}",
                expected=expected,
                actual=actual,
            )
        return self.ok(actual=actual)


@CHECKS.register("state_unchanged")
class StateUnchanged(_TranscriptCheck):
    """A path in the system of record is the same at the end as at the start.

    args: ``path``. Needs a snapshot with ``initial`` and ``final`` sections (see the module
    docstring), e.g. a balance that must not move until a request is approved.
    """

    def check(self, transcript: Transcript) -> CheckResult:
        path: str = self.args["path"]
        snapshot = transcript.final_state
        if "initial" not in snapshot or "final" not in snapshot:
            return self.fail("snapshot has no initial/final sections to compare")
        found_before, before = lookup(snapshot["initial"], path)
        found_after, after = lookup(snapshot["final"], path)
        if not found_before and not found_after:
            return self.fail(f"{path} not found in the snapshot")
        if found_before != found_after or not same_value(after, before):
            return self.fail(
                f"{path} changed from {before!r} to {after!r}",
                before=before if found_before else None,
                after=after if found_after else None,
            )
        return self.ok(value=after)


# ----------------------------------------------------------------------------- what was said


@CHECKS.register("no_disclosure")
class NoDisclosure(_TranscriptCheck):
    """None of the given values appears in any agent message: a deterministic leak sensor.

    args: ``values: [...]`` (e.g. another employee's salary in each spelling worth catching).
    Matching is case- and whitespace-insensitive. Tool results are not scanned: the agent
    *reading* data it is allowed to read is not a leak; *saying* it is.
    """

    def check(self, transcript: Transcript) -> CheckResult:
        values = [str(v) for v in self.args.get("values", [])]
        leaks: list[tuple[int, str]] = [
            (i, v)
            for i, text in _agent_messages(transcript)
            for v in values
            if norm_text(v) and norm_text(v) in norm_text(text)
        ]
        if leaks:
            shown = sorted({v for _, v in leaks})
            return self.fail(
                f"agent disclosed {', '.join(shown)}",
                leaks=[{"message": i, "value": v} for i, v in leaks],
            )
        return self.ok(values=len(values))


@CHECKS.register("no_false_success")
class NoFalseSuccess(_TranscriptCheck):
    """The agent never claims a write that did not commit ("hallucination is a missing sensor").

    args: ``tool`` (the write tool), ``claim_pattern`` (case-insensitive regex for a success
    claim, e.g. ``has been submitted|all set``).

    If at least one call to ``tool`` succeeded, any claim is allowed. Otherwise no agent message
    may match ``claim_pattern``: after the first attempted call if there was one (so quoting the
    change *before* writing is fine), or anywhere if the tool was never called at all.
    """

    def check(self, transcript: Transcript) -> CheckResult:
        tool: str = self.args["tool"]
        pattern = re.compile(self.args["claim_pattern"], flags=re.IGNORECASE)
        calls = [e for e in transcript.tool_events if e.call.name == tool]
        if any(e.error is None for e in calls):
            return self.ok(committed=True)
        start = 0
        if calls:
            first_id = calls[0].call.id
            start = next(
                (
                    i
                    for i, m in enumerate(transcript.messages)
                    if any(c.id == first_id for c in m.tool_calls)
                ),
                0,
            )
        claims = [
            {"message": i, "claim": match.group(0), "text": text}
            for i, text in _agent_messages(transcript)
            if i >= start and (match := pattern.search(text)) is not None
        ]
        if claims:
            why = "every call failed" if calls else "it was never called"
            return self.fail(
                f"agent claimed success ({claims[0]['claim']!r}) but {tool} did not commit ({why})",
                claims=claims,
                errors=[e.error for e in calls],
            )
        return self.ok(committed=False, attempts=len(calls))


@CHECKS.register("termination_in")
class TerminationIn(_TranscriptCheck):
    """The conversation ended for one of the allowed reasons.

    args: ``allowed: [goal_met, agent_ended, user_gave_up, ...]``.
    """

    def check(self, transcript: Transcript) -> CheckResult:
        allowed = [str(a) for a in self.args.get("allowed", [])]
        reason = transcript.termination.value
        if reason not in allowed:
            return self.fail(f"terminated with {reason}, allowed: {', '.join(allowed)}")
        return self.ok(termination=reason)


__all__ = ["contains_subset", "same_value", "step_score"]
