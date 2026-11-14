"""Judge calibration: treat the LLM judge as a classifier and measure it against people.

``positive`` means "pass". On a human-labelled sample:

* TPR: P(judge pass | human pass); TNR: P(judge fail | human fail).
* Cohen's kappa: chance-corrected agreement.
* Corrected pass rate (Rogan-Gladen): the observed judge pass rate p is biased by the judge's
  errors; the true rate is estimated as (p + TNR - 1) / (TPR + TNR - 1), clipped to [0, 1].
  Only meaningful when the judge beats chance (TPR + TNR > 1).
"""

from __future__ import annotations

from collections.abc import Sequence

from assayer.domain.base import Model
from assayer.domain.gold import cohens_kappa
from assayer.domain.judging import Verdict


class CalibrationStats(Model):
    n: int
    agreement: float | None
    kappa: float | None
    tpr: float | None
    tnr: float | None
    human_positives: int
    human_negatives: int

    def corrected_rate(self, observed_pass_rate: float) -> float | None:
        if self.tpr is None or self.tnr is None or self.tpr + self.tnr <= 1:
            return None
        estimate = (observed_pass_rate + self.tnr - 1) / (self.tpr + self.tnr - 1)
        return min(1.0, max(0.0, estimate))


def calibration_stats(pairs: Sequence[tuple[Verdict, Verdict]]) -> CalibrationStats:
    """``pairs`` are ``(judge_verdict, human_verdict)``."""
    pos = [(j, h) for j, h in pairs if h == "pass"]
    neg = [(j, h) for j, h in pairs if h == "fail"]
    return CalibrationStats(
        n=len(pairs),
        agreement=sum(j == h for j, h in pairs) / len(pairs) if pairs else None,
        kappa=cohens_kappa(list(pairs)) if pairs else None,
        tpr=sum(j == "pass" for j, _ in pos) / len(pos) if pos else None,
        tnr=sum(j == "fail" for j, _ in neg) / len(neg) if neg else None,
        human_positives=len(pos),
        human_negatives=len(neg),
    )
