"""Checks for single-step extraction outputs against gold labels."""

from __future__ import annotations

from typing import Any

from assayer.application.registry import CHECKS
from assayer.builtin.checks.base import Case, Output, SpecCheck, lookup, norm_text, norm_value
from assayer.domain.checks import CheckResult
from assayer.domain.errors import UnresolvedGold
from assayer.domain.fixtures import TaskFixture
from assayer.domain.outputs import TaskOutput
from assayer.domain.quotes import verify_quote


def _task_parts(case: Case, output: Output) -> tuple[TaskFixture, dict[str, Any]] | None:
    if not isinstance(case, TaskFixture) or not isinstance(output, TaskOutput):
        return None
    return case, output.parsed or {}


@CHECKS.register("field_match")
class FieldMatch(SpecCheck):
    """Every listed gold field (default: all) equals the output, compared leniently.

    args: ``fields: [name, ...]``. Strings are compared case-, whitespace- and trailing-dot-
    insensitively, numbers numerically, and lists as sets.
    """

    def evaluate(self, case: Case, output: Output) -> CheckResult:
        parts = _task_parts(case, output)
        if parts is None:
            return self.fail("field_match applies to task outputs only")
        fixture, parsed = parts
        fields: list[str] = self.args.get("fields") or sorted(fixture.gold)
        mismatches: dict[str, Any] = {}
        skipped: list[str] = []
        for name in fields:
            gold = fixture.gold.get(name)
            if gold is None:
                return self.fail(f"no gold label for field {name!r}")
            try:
                expected = gold.expected
            except UnresolvedGold:
                skipped.append(name)
                continue
            found, actual = lookup(parsed, name)
            if not found or norm_value(actual) != norm_value(expected):
                mismatches[name] = {"expected": expected, "actual": actual if found else None}
        if mismatches:
            return self.fail(
                f"{len(mismatches)} field(s) differ: {', '.join(mismatches)}",
                mismatches=mismatches,
                skipped=skipped,
            )
        return self.ok(fields=len(fields) - len(skipped), skipped=skipped)


@CHECKS.register("set_f1")
class SetF1(SpecCheck):
    """F1 between an output list and the gold list, at or above ``threshold`` (default 0.8)."""

    def evaluate(self, case: Case, output: Output) -> CheckResult:
        parts = _task_parts(case, output)
        if parts is None:
            return self.fail("set_f1 applies to task outputs only")
        fixture, parsed = parts
        field: str = self.args["field"]
        threshold = float(self.args.get("threshold", 0.8))
        gold = fixture.gold.get(field)
        if gold is None:
            return self.fail(f"no gold label for field {field!r}")
        expected = {norm_text(v) for v in (gold.expected or [])}  # type: ignore[union-attr]
        found, actual_raw = lookup(parsed, field)
        actual = (
            {norm_text(v) for v in actual_raw} if found and isinstance(actual_raw, list) else set()
        )
        tp = len(expected & actual)
        precision = tp / len(actual) if actual else 0.0
        recall = tp / len(expected) if expected else 1.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        detail = {
            "f1": round(f1, 4),
            "missing": sorted(expected - actual),
            "extra": sorted(actual - expected),
        }
        if f1 >= threshold:
            return self.ok(**detail)
        return self.fail(f"F1 {f1:.2f} < {threshold:.2f} for {field}", **detail)


@CHECKS.register("trap_absent")
class TrapAbsent(SpecCheck):
    """No output field equals one of its gold *trap* values (a known plausible-but-wrong answer).

    Hitting a trap is a more specific diagnosis than a plain mismatch: it says *which* mistake
    the model made (e.g. it reported the previous employer as the current one).
    """

    def evaluate(self, case: Case, output: Output) -> CheckResult:
        parts = _task_parts(case, output)
        if parts is None:
            return self.fail("trap_absent applies to task outputs only")
        fixture, parsed = parts
        hits = []
        for name, gold in fixture.gold.items():
            found, actual = lookup(parsed, name)
            if found and any(norm_value(actual) == norm_value(t) for t in gold.traps):
                hits.append({"field": name, "value": actual})
        if hits:
            names = ", ".join(h["field"] for h in hits)
            return self.fail(f"trap value produced for {names}", hits=hits)
        return self.ok()


@CHECKS.register("grounded")
class Grounded(SpecCheck):
    """Every string in the listed output fields appears in the input document.

    args: ``fields: [...]``, ``min_chars`` (default 2). Catches fabricated values that a gold
    comparison cannot (for fields without gold, or free-text fields).
    """

    def evaluate(self, case: Case, output: Output) -> CheckResult:
        parts = _task_parts(case, output)
        if parts is None:
            return self.fail("grounded applies to task outputs only")
        fixture, parsed = parts
        min_chars = int(self.args.get("min_chars", 2))
        ungrounded: list[str] = []
        for name in self.args.get("fields", []):
            found, value = lookup(parsed, name)
            if not found or value is None:
                continue
            values = value if isinstance(value, list) else [value]
            ungrounded.extend(
                f"{name}={v!r}"
                for v in values
                if isinstance(v, str)
                and not verify_quote(v, fixture.input_doc, min_chars=min_chars)
            )
        if ungrounded:
            return self.fail(f"not found in source: {'; '.join(ungrounded)}", ungrounded=ungrounded)
        return self.ok()
