from __future__ import annotations

import pytest

from assayer.application.calibration import judge_vs_human, labels_from_items, stratified_sample
from assayer.application.compare import compare
from assayer.domain.attempt import Attempt, AttemptStatus
from assayer.domain.calibration import calibration_stats
from assayer.domain.errors import ErrorKind
from assayer.domain.grading import Grade
from assayer.domain.judging import JudgeResponse, judge_error, settle_verdict
from assayer.domain.outcome import Outcome
from assayer.domain.stats import mcnemar_exact
from assayer.ports.store import CalibrationLabel, StoredJudgement


def _graded(outcomes: dict[str, Outcome]) -> list[tuple[Attempt, Grade]]:
    return [
        (
            Attempt(
                attempt_id=i,
                run_id="r",
                case_id=case,
                repeat=0,
                status=AttemptStatus.SUCCEEDED,
            ),
            Grade(outcome=o),
        )
        for i, (case, o) in enumerate(outcomes.items())
    ]


P, F, E = Outcome.PASS, Outcome.FAIL, Outcome.HARNESS_ERROR


def test_mcnemar_known_values() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(1, 9) == pytest.approx(0.021484375)


def test_compare_pairs_only_graded_cases_on_both_sides() -> None:
    base = _graded({"a": P, "b": P, "c": F, "d": P, "e": E})
    cand = _graded({"a": P, "b": F, "c": P, "d": E, "e": P, "z": P})
    c = compare("base", base, "cand", cand)
    assert c.pairs == 3  # a, b, c (d and e have a harness error on one side)
    assert (c.regressions, c.improvements) == (1, 1)
    assert c.paired_delta == 0.0
    assert [f.key for f in c.flips] == ["b#0", "c#0"]
    assert c.unpaired == 1  # z
    assert not c.regressed(0.0)


def test_regression_gate() -> None:
    base = _graded(dict.fromkeys("abcdefghij", P))
    cand = _graded({**dict.fromkeys("abcdefgh", P), "i": F, "j": F})
    c = compare("base", base, "cand", cand)
    assert c.paired_delta == pytest.approx(-0.2)
    assert c.regressed(0.1)
    assert not c.regressed(0.25)


def test_calibration_stats_and_rogan_gladen_correction() -> None:
    # 10 human passes: judge says pass on 9. 10 human fails: judge says fail on 8.
    pairs = (
        [("pass", "pass")] * 9
        + [("fail", "pass")]
        + [("fail", "fail")] * 8
        + [("pass", "fail")] * 2
    )
    stats = calibration_stats(pairs)  # type: ignore[arg-type]
    assert stats.tpr == pytest.approx(0.9)
    assert stats.tnr == pytest.approx(0.8)
    assert stats.agreement == pytest.approx(0.85)
    # Observed 0.6 judge-pass rate corrects to (0.6 + 0.8 - 1) / (0.9 + 0.8 - 1) = 0.571...
    assert stats.corrected_rate(0.6) == pytest.approx(0.4 / 0.7)
    assert calibration_stats([("pass", "fail"), ("fail", "pass")]).corrected_rate(0.5) is None


def _judgement(attempt_id: int, verdict: str) -> StoredJudgement:
    if verdict == "error":
        v = judge_error(ErrorKind.JUDGE_INCONSISTENT, "x")
    else:
        v = settle_verdict(JudgeResponse(reasoning="r", verdict="pass"), "")
        if verdict == "fail":
            v = v.model_copy(update={"verdict_final": "fail"})
    return StoredJudgement(
        attempt_id=attempt_id,
        judge_id="llm_judge",
        judge_version="v",
        evidence_hash="h",
        judge_model="m",
        verdict=v,
    )


def test_stratified_sample_balances_verdicts_and_skips_errors() -> None:
    judgements = (
        [_judgement(i, "pass") for i in range(10)]
        + [_judgement(100 + i, "fail") for i in range(2)]
        + [_judgement(200, "error")]
    )
    sample = stratified_sample(judgements, 4)
    verdicts = [j.verdict.verdict_final for j in sample]
    assert verdicts.count("fail") == 2
    assert verdicts.count("pass") == 2
    assert stratified_sample(judgements, 4) == sample  # seeded, reproducible


def test_labels_import_ignores_unlabelled_items_and_joins_by_attempt() -> None:
    items = [
        {"attempt_id": 1, "judge_version": "v", "human_verdict": "pass"},
        {"attempt_id": 2, "judge_version": "v", "human_verdict": None},
        {"attempt_id": 3, "judge_version": "v", "human_verdict": "fail", "note": "made up"},
    ]
    labels = labels_from_items(items, "ann")
    assert [label.attempt_id for label in labels] == [1, 3]
    assert labels[1] == CalibrationLabel(
        attempt_id=3,
        judge_id="llm_judge",
        judge_version="v",
        labeller="ann",
        human_verdict="fail",
        note="made up",
    )
    stats = judge_vs_human([_judgement(1, "pass"), _judgement(3, "pass")], labels)
    assert stats.n == 2
    assert stats.tnr == 0.0  # the judge passed something people failed
