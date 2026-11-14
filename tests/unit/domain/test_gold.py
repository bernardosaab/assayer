import math

import pytest

from assayer.domain.errors import UnresolvedGold
from assayer.domain.gold import GoldField, cohens_kappa, percent_agreement


def test_shorthand_value_is_resolved() -> None:
    field = GoldField.model_validate("Acme Corp")
    assert field.expected == "Acme Corp"
    assert field.is_resolved


def test_unanimous_votes_are_expected() -> None:
    field = GoldField(votes={"ann": 7, "bo": 7})
    assert field.expected == 7


def test_disagreeing_votes_without_resolution_raise() -> None:
    field = GoldField(votes={"ann": 7, "bo": 8})
    assert not field.is_resolved
    with pytest.raises(UnresolvedGold, match="disagree"):
        _ = field.expected


def test_resolution_breaks_tie() -> None:
    assert GoldField(votes={"ann": 7, "bo": 8}, resolved=8).expected == 8


def test_open_dispute_blocks_even_with_resolution() -> None:
    field = GoldField(votes={"a": 1, "b": 2}, resolved=2, dispute={"assignee": "lead"})
    with pytest.raises(UnresolvedGold, match="lead"):
        _ = field.expected


def test_vote_comparison_is_structural() -> None:
    assert GoldField(votes={"a": ["x", "y"], "b": ["x", "y"]}).is_unanimous


def test_percent_agreement() -> None:
    assert percent_agreement([("a", "a"), ("a", "b")]) == 0.5
    assert math.isnan(percent_agreement([]))


def test_cohens_kappa_textbook_example() -> None:
    # Classic 2x2: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> kappa = 0.4
    pairs = [("y", "y")] * 20 + [("y", "n")] * 5 + [("n", "y")] * 10 + [("n", "n")] * 15
    assert cohens_kappa(pairs) == pytest.approx(0.4)


def test_cohens_kappa_degenerate_cases() -> None:
    assert cohens_kappa([("a", "a"), ("a", "a")]) == 1.0
    assert math.isnan(cohens_kappa([]))
