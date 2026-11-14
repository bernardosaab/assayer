"""Judge calibration workflow: sample -> people label blind -> import -> stats.

The sample is stratified by the judge's verdict so both classes are represented even when one
dominates, and the export is *blind*: labellers see the evidence and criteria, never the judge's
verdict, so they cannot anchor on it.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from assayer.application.scoring import JUDGE_ID
from assayer.domain.calibration import CalibrationStats, calibration_stats
from assayer.domain.judging import Verdict
from assayer.ports.store import CalibrationLabel, StoredJudgement


def stratified_sample(
    judgements: list[StoredJudgement], n: int, *, seed: int = 0
) -> list[StoredJudgement]:
    """Up to ``n`` finished judgements, split as evenly as possible across verdicts."""
    by_verdict: dict[str, list[StoredJudgement]] = defaultdict(list)
    for j in judgements:
        if j.verdict.verdict_final is not None:
            by_verdict[j.verdict.verdict_final].append(j)
    rng = random.Random(seed)  # noqa: S311 - sampling, not crypto
    for group in by_verdict.values():
        rng.shuffle(group)
    picked: list[StoredJudgement] = []
    groups = [by_verdict[k] for k in sorted(by_verdict)]
    while len(picked) < n and any(groups):
        for group in groups:
            if group and len(picked) < n:
                picked.append(group.pop())
    return sorted(picked, key=lambda j: j.attempt_id)


def blind_item(
    j: StoredJudgement, case_id: str, evidence: str, criteria: list[str]
) -> dict[str, Any]:
    return {
        "attempt_id": j.attempt_id,
        "case_id": case_id,
        "judge_version": j.judge_version,
        "criteria": criteria,
        "evidence": evidence,
        "human_verdict": None,  # fill in: pass | fail
        "note": "",
    }


def labels_from_items(items: list[dict[str, Any]], labeller: str) -> list[CalibrationLabel]:
    labels = []
    for item in items:
        verdict = item.get("human_verdict")
        if verdict not in ("pass", "fail"):
            continue
        labels.append(
            CalibrationLabel(
                attempt_id=int(item["attempt_id"]),
                judge_id=JUDGE_ID,
                judge_version=str(item["judge_version"]),
                labeller=labeller,
                human_verdict=verdict,
                note=str(item.get("note") or ""),
            )
        )
    return labels


def judge_vs_human(
    judgements: list[StoredJudgement], labels: list[CalibrationLabel]
) -> CalibrationStats:
    by_attempt: dict[int, Verdict] = {
        j.attempt_id: j.verdict.verdict_final
        for j in judgements
        if j.verdict.verdict_final is not None
    }
    pairs = [
        (by_attempt[label.attempt_id], label.human_verdict)
        for label in labels
        if label.attempt_id in by_attempt
    ]
    return calibration_stats(pairs)
