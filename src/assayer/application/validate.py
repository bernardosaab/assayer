"""Static validation of an eval before spending anything: ``assayer validate``.

Catches what would otherwise surface mid-run or, worse, silently skew results: unknown checks,
unresolved gold labels, trap values equal to the expected value, criteria without text.
"""

from __future__ import annotations

from pydantic import Field

from assayer.application.registry import build_check
from assayer.domain.base import Model, canonical_json
from assayer.domain.errors import ConfigError, UnresolvedGold
from assayer.domain.evalspec import EvalKind, EvalSpec
from assayer.domain.fixtures import ConversationFixture, TaskFixture


class Issue(Model):
    severity: str  # "error" | "warning"
    where: str
    message: str


class ValidationReport(Model):
    eval_name: str
    cases: int
    issues: list[Issue] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)


def validate_eval(spec: EvalSpec) -> ValidationReport:
    issues: list[Issue] = []

    def err(where: str, message: str) -> None:
        issues.append(Issue(severity="error", where=where, message=message))

    def warn(where: str, message: str) -> None:
        issues.append(Issue(severity="warning", where=where, message=message))

    if spec.kind is EvalKind.TASK and spec.task is None:
        err("eval.yaml", "kind 'task' needs a 'task' section")
    if spec.kind is EvalKind.CONVERSATION and (spec.agent is None or spec.simulator is None):
        err("eval.yaml", "kind 'conversation' needs 'agent' and 'simulator' sections")

    for case in spec.cases:
        issues.extend(_case_issues(spec, case))
    if spec.fake_script is None:
        warn("eval", "no fake/script.yaml; this eval cannot run offline (--profile fake)")
    return ValidationReport(eval_name=spec.name, cases=len(spec.cases), issues=issues)


def _case_issues(spec: EvalSpec, case: TaskFixture | ConversationFixture) -> list[Issue]:
    where = f"case {case.id}"
    issues: list[Issue] = []
    check_ids = [c.check_id for c in spec.checks_for(case)]
    duplicates = sorted({i for i in check_ids if check_ids.count(i) > 1})
    if duplicates:
        issues.append(
            Issue(
                severity="error",
                where=where,
                message=f"duplicate check ids {duplicates}; give each an explicit `id`",
            )
        )
    for check_spec in spec.checks_for(case):
        try:
            build_check(check_spec)
        except ConfigError as exc:
            issues.append(Issue(severity="error", where=where, message=str(exc)))
    ids = [c.id for c in spec.criteria_for(case)]
    if len(ids) != len(set(ids)):
        issues.append(Issue(severity="error", where=where, message=f"duplicate criteria: {ids}"))
    if ids and spec.judge is None:
        issues.append(
            Issue(severity="warning", where=where, message="criteria given but no judge defined")
        )
    if isinstance(case, TaskFixture):
        issues.extend(_gold_issues(where, case))
    return issues


def _gold_issues(where: str, case: TaskFixture) -> list[Issue]:
    issues: list[Issue] = []
    for name, gold in case.gold.items():
        at = f"{where} gold.{name}"
        try:
            expected = gold.expected
        except UnresolvedGold as exc:
            issues.append(Issue(severity="error", where=at, message=f"unresolved: {exc}"))
            continue
        if any(canonical_json(t) == canonical_json(expected) for t in gold.traps):
            issues.append(
                Issue(severity="error", where=at, message="a trap equals the expected value")
            )
        if len(gold.votes) == 1:
            issues.append(
                Issue(severity="warning", where=at, message="one labeller vote; agreement unknown")
            )
    return issues
