"""Combine an attempt's execution status, check results and judge verdict into one outcome."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from assayer.domain.attempt import AttemptStatus
from assayer.domain.base import Model
from assayer.domain.checks import CheckResult
from assayer.domain.judging import JudgeStatus, JudgeVerdict
from assayer.domain.outcome import Outcome

GRADER_VERSION = "1"


class Grade(Model):
    outcome: Outcome
    reasons: list[str] = Field(default_factory=list)


def grade(
    *,
    attempt_status: AttemptStatus,
    agent_error: str | None,
    checks: Sequence[CheckResult],
    judge: JudgeVerdict | None,
) -> Grade:
    """Precedence: harness errors first (we could not measure), then any failure reason.

    * The attempt or its judgement hit infrastructure trouble -> ``harness_error``.
    * Agent error, failed check, or verified judge finding -> ``fail`` (all reasons listed).
    * Otherwise -> ``pass``.
    """
    if attempt_status is AttemptStatus.HARNESS_ERROR:
        return Grade(outcome=Outcome.HARNESS_ERROR, reasons=["attempt: harness error"])
    if attempt_status is not AttemptStatus.SUCCEEDED:
        raise ValueError(f"cannot grade an attempt that is {attempt_status.value}")
    if judge is not None and judge.status is JudgeStatus.HARNESS_ERROR:
        kind = judge.error_kind.value if judge.error_kind else "unknown"
        return Grade(outcome=Outcome.HARNESS_ERROR, reasons=[f"judge: {kind}"])

    reasons: list[str] = []
    if agent_error:
        reasons.append(f"agent: {agent_error}")
    reasons.extend(
        f"check {c.check_id}: {c.detail.get('reason', 'failed')}" for c in checks if not c.passed
    )
    if judge is not None and judge.verdict_final == "fail":
        reasons.extend(
            f"judge {f.finding.criterion_id}: {f.finding.explanation or f.finding.quote}"
            for f in judge.findings
            if f.verified
        )
    return Grade(outcome=Outcome.FAIL if reasons else Outcome.PASS, reasons=reasons)
