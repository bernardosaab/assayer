"""Gold labels: expected values with labeller votes, disputes and trap values (ADR-0016).

A gold field records *how* the expected value was decided, not only what it is:

* ``votes``: one value per labeller (normally two).
* ``resolved``: a tiebreak decision when votes disagree.
* ``dispute``: an open disagreement assigned to a person. An open dispute blocks the case.
* ``traps``: plausible-but-wrong values (e.g. a *previous* employer for ``current_employer``).
  Producing a trap is a more specific failure than producing any other wrong value.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from pydantic import Field, JsonValue, model_validator

from assayer.domain.base import Model, canonical_json
from assayer.domain.errors import UnresolvedGold


class Dispute(Model):
    assignee: str
    note: str = ""


class GoldField(Model):
    votes: dict[str, JsonValue] = Field(default_factory=dict)
    resolved: JsonValue | None = None
    dispute: Dispute | None = None
    traps: list[JsonValue] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, data: Any) -> Any:
        """Allow ``field: value`` in YAML as shorthand for an already-resolved gold value."""
        if isinstance(data, GoldField) or (
            isinstance(data, dict) and set(data) & {"votes", "resolved", "dispute", "traps"}
        ):
            return data
        return {"resolved": data}

    @property
    def is_unanimous(self) -> bool:
        return len({canonical_json(v) for v in self.votes.values()}) <= 1

    @property
    def is_resolved(self) -> bool:
        if self.dispute is not None:
            return False
        return self.resolved is not None or (bool(self.votes) and self.is_unanimous)

    @property
    def expected(self) -> JsonValue:
        """The value an output must match.

        Raises:
            UnresolvedGold: if there is an open dispute, or votes disagree with no resolution.
        """
        if self.dispute is not None:
            raise UnresolvedGold(f"open dispute assigned to {self.dispute.assignee}")
        if self.resolved is not None:
            return self.resolved
        if self.votes and self.is_unanimous:
            return next(iter(self.votes.values()))
        raise UnresolvedGold(f"labellers disagree: {self.votes!r}")


def percent_agreement(pairs: Sequence[tuple[JsonValue, JsonValue]]) -> float:
    """Share of items on which two labellers gave the same value."""
    if not pairs:
        return float("nan")
    same = sum(canonical_json(a) == canonical_json(b) for a, b in pairs)
    return same / len(pairs)


def cohens_kappa(pairs: Sequence[tuple[JsonValue, JsonValue]]) -> float:
    """Cohen's kappa for two raters over categorical labels (values compared canonically).

    kappa = (p_o - p_e) / (1 - p_e). Returns 1.0 when both raters use one identical label
    throughout (p_e == 1), and NaN for no data.
    """
    if not pairs:
        return float("nan")
    n = len(pairs)
    a_labels = [canonical_json(a) for a, _ in pairs]
    b_labels = [canonical_json(b) for _, b in pairs]
    p_o = sum(x == y for x, y in zip(a_labels, b_labels, strict=True)) / n
    a_counts, b_counts = Counter(a_labels), Counter(b_labels)
    p_e = sum((a_counts[k] / n) * (b_counts[k] / n) for k in a_counts.keys() | b_counts.keys())
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)
