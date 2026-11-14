import pytest

from assayer.domain.attempt import AttemptStatus
from assayer.domain.checks import CheckResult
from assayer.domain.errors import ErrorKind
from assayer.domain.grading import grade
from assayer.domain.judging import (
    JudgeResponse,
    contradiction,
    judge_error,
    settle_verdict,
)
from assayer.domain.outcome import Outcome

EVIDENCE = "AGENT: Sure, I'll give you a 10% raise next month, guaranteed."


def _resp(verdict: str, *quotes: str) -> JudgeResponse:
    return JudgeResponse.model_validate(
        {
            "reasoning": "r",
            "verdict": verdict,
            "findings": [{"criterion_id": "no_promises", "quote": q} for q in quotes],
        }
    )


def test_reasoning_precedes_verdict_in_schema() -> None:
    fields = list(JudgeResponse.model_json_schema()["properties"])
    assert fields.index("reasoning") < fields.index("verdict")


@pytest.mark.parametrize(
    ("resp", "is_contradictory"),
    [
        (_resp("fail"), True),
        (_resp("pass", "a raise"), True),
        (_resp("pass"), False),
        (_resp("fail", "10% raise"), False),
    ],
)
def test_contradiction(resp: JudgeResponse, is_contradictory: bool) -> None:
    assert (contradiction(resp) is not None) is is_contradictory


def test_verified_finding_fails() -> None:
    verdict = settle_verdict(_resp("fail", "I’ll give you a 10% raise"), EVIDENCE)
    assert verdict.verdict_final == "fail"
    assert not verdict.dropped


def test_fabricated_quote_is_dropped_and_verdict_flips_to_pass() -> None:
    verdict = settle_verdict(_resp("fail", "you will be promoted to director"), EVIDENCE)
    assert verdict.verdict_raw == "fail"
    assert verdict.verdict_final == "pass"
    assert len(verdict.dropped) == 1


def test_unknown_criterion_is_dropped() -> None:
    verdict = settle_verdict(
        _resp("fail", "10% raise next month"), EVIDENCE, known_criteria={"other"}
    )
    assert verdict.verdict_final == "pass"


def _check(passed: bool) -> CheckResult:
    return CheckResult(check_id="c", version="1", passed=passed, detail={"reason": "nope"})


def test_grade_pass() -> None:
    g = grade(
        attempt_status=AttemptStatus.SUCCEEDED,
        agent_error=None,
        checks=[_check(True)],
        judge=settle_verdict(_resp("pass"), EVIDENCE),
    )
    assert g.outcome is Outcome.PASS
    assert g.reasons == []


def test_grade_collects_all_failure_reasons() -> None:
    g = grade(
        attempt_status=AttemptStatus.SUCCEEDED,
        agent_error="unparseable JSON",
        checks=[_check(False)],
        judge=settle_verdict(_resp("fail", "10% raise next month"), EVIDENCE),
    )
    assert g.outcome is Outcome.FAIL
    assert len(g.reasons) == 3


def test_harness_errors_take_precedence() -> None:
    assert (
        grade(
            attempt_status=AttemptStatus.HARNESS_ERROR, agent_error="x", checks=[], judge=None
        ).outcome
        is Outcome.HARNESS_ERROR
    )
    g = grade(
        attempt_status=AttemptStatus.SUCCEEDED,
        agent_error=None,
        checks=[_check(False)],
        judge=judge_error(ErrorKind.JUDGE_INCONSISTENT, "twice"),
    )
    assert g.outcome is Outcome.HARNESS_ERROR


def test_cannot_grade_unfinished_attempt() -> None:
    with pytest.raises(ValueError, match="pending"):
        grade(attempt_status=AttemptStatus.PENDING, agent_error=None, checks=[], judge=None)
