from decimal import Decimal

import pytest

from assayer.domain.money import Money


def test_from_usd_rounds_up_to_never_undercount() -> None:
    assert Money.from_usd("0.0000001") == Money(1)
    assert Money.from_usd("1.25") == Money(1_250_000)


def test_arithmetic_and_ordering() -> None:
    a, b = Money(300), Money(200)
    assert a + b == Money(500)
    assert a - b == Money(100)
    assert b * 3 == Money(600)
    assert b < a
    assert sum([a, b], Money.zero()) == Money(500)


def test_round_trip_and_str() -> None:
    assert Money.from_usd(Decimal("2.5")).to_usd() == Decimal("2.5")
    assert str(Money(1_500_000)) == "$1.500000"


@pytest.mark.parametrize("bad", [1.5, True, "3"])
def test_rejects_non_int(bad: object) -> None:
    with pytest.raises(TypeError):
        Money(bad)  # type: ignore[arg-type]
