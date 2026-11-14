"""The judge protocol (ADR-0010).

The judge is part of the experiment and is checked like the agent:

1. **Reasoning before verdict.** :class:`JudgeResponse` declares ``reasoning`` first, so a
   structured-output model writes its reasoning before committing to a verdict.
2. **Contradictions are re-asked.** A ``fail`` with no findings, or a ``pass`` that lists
   violations, is internally inconsistent. The caller re-asks once, and a second contradiction
   is a harness error.
3. **Every quote is verified.** Findings whose quote is not verbatim in the evidence are dropped.
4. **The final verdict comes from verified findings**, not from the model's own verdict word.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

from pydantic import Field

from assayer.domain.base import Model
from assayer.domain.errors import ErrorKind
from assayer.domain.quotes import DEFAULT_MIN_CHARS, verify_quote

Verdict = Literal["pass", "fail"]


class Finding(Model):
    """One violation of one criterion, backed by a verbatim quote."""

    criterion_id: str
    quote: str
    explanation: str = ""


class JudgeResponse(Model):
    """The structured output requested from the judge model. Field order is significant."""

    reasoning: str = Field(description="Think step by step about each criterion first.")
    findings: list[Finding] = Field(
        default_factory=list,
        description="One entry per violated criterion, each quoting the evidence verbatim.",
    )
    verdict: Verdict = Field(description="'fail' if and only if findings is non-empty.")


def contradiction(response: JudgeResponse) -> str | None:
    """Describe the internal inconsistency in a judge response, or return None."""
    if response.verdict == "fail" and not response.findings:
        return "verdict is 'fail' but no findings were given"
    if response.verdict == "pass" and response.findings:
        return "verdict is 'pass' but findings list violations"
    return None


class CheckedFinding(Model):
    finding: Finding
    verified: bool


class JudgeStatus(StrEnum):
    DONE = "done"
    HARNESS_ERROR = "harness_error"


class JudgeVerdict(Model):
    status: JudgeStatus
    verdict_final: Verdict | None = None
    verdict_raw: Verdict | None = None
    reasoning: str = ""
    findings: tuple[CheckedFinding, ...] = ()
    reasks: int = 0
    error_kind: ErrorKind | None = None
    error_msg: str | None = None

    @property
    def dropped(self) -> tuple[CheckedFinding, ...]:
        return tuple(f for f in self.findings if not f.verified)


def settle_verdict(
    response: JudgeResponse,
    evidence: str,
    *,
    reasks: int = 0,
    known_criteria: set[str] | None = None,
    min_chars: int = DEFAULT_MIN_CHARS,
    sources: Mapping[str, str] | None = None,
) -> JudgeVerdict:
    """Verify each finding's quote and derive the final verdict.

    A finding is kept only if its quote verifies against its criterion's source (``sources``
    maps criterion id to the text it must quote; default ``evidence``) and, when given, it names
    a known criterion.
    """
    sources = sources or {}
    checked = tuple(
        CheckedFinding(
            finding=f,
            verified=verify_quote(
                f.quote, sources.get(f.criterion_id, evidence), min_chars=min_chars
            )
            and (known_criteria is None or f.criterion_id in known_criteria),
        )
        for f in response.findings
    )
    final: Verdict = "fail" if any(c.verified for c in checked) else "pass"
    return JudgeVerdict(
        status=JudgeStatus.DONE,
        verdict_final=final,
        verdict_raw=response.verdict,
        reasoning=response.reasoning,
        findings=checked,
        reasks=reasks,
    )


def judge_error(kind: ErrorKind, message: str, *, reasks: int = 0) -> JudgeVerdict:
    return JudgeVerdict(
        status=JudgeStatus.HARNESS_ERROR, error_kind=kind, error_msg=message, reasks=reasks
    )
