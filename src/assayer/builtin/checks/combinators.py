"""Composite checks (Composite/Specification pattern): ``all_of``, ``any_of``, ``not``.

```yaml
- use: any_of
  id: books_or_escalates
  args:
    checks:
      - {use: tool_called, args: {name: submit_pto_request}}
      - {use: tool_called, args: {name: escalate_to_hr}}
```
"""

from __future__ import annotations

from functools import cached_property

from assayer.application.registry import CHECKS, build_check
from assayer.builtin.checks.base import Case, Output, SpecCheck
from assayer.domain.checks import CheckResult
from assayer.domain.fixtures import CheckSpec
from assayer.ports.grading import Check


class _Composite(SpecCheck):
    @cached_property
    def children(self) -> list[Check]:
        return [build_check(CheckSpec.model_validate(c)) for c in self.args.get("checks", [])]

    def _results(self, case: Case, output: Output) -> list[CheckResult]:
        return [child.evaluate(case, output) for child in self.children]

    @staticmethod
    def _summary(results: list[CheckResult]) -> list[dict[str, object]]:
        return [{"check": r.check_id, "passed": r.passed, **r.detail} for r in results]


@CHECKS.register("all_of")
class AllOf(_Composite):
    def evaluate(self, case: Case, output: Output) -> CheckResult:
        results = self._results(case, output)
        failed = [r.check_id for r in results if not r.passed]
        if failed:
            return self.fail(f"failed: {', '.join(failed)}", parts=self._summary(results))
        return self.ok(parts=self._summary(results))


@CHECKS.register("any_of")
class AnyOf(_Composite):
    def evaluate(self, case: Case, output: Output) -> CheckResult:
        results = self._results(case, output)
        if any(r.passed for r in results):
            return self.ok(parts=self._summary(results))
        return self.fail("none of the alternatives passed", parts=self._summary(results))


@CHECKS.register("not")
class Not(_Composite):
    def evaluate(self, case: Case, output: Output) -> CheckResult:
        [result] = self._results(case, output)
        if result.passed:
            return self.fail(f"{result.check_id} passed but must not")
        return self.ok()
