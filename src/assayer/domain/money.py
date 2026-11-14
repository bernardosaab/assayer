"""Exact money arithmetic in integer micro-dollars (ADR-0007).

Floats accumulate error when summing thousands of tiny per-call costs, and a budget check must
be exact, so all money is an integer count of millionths of a dollar.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal

_MICROS_PER_USD = Decimal(1_000_000)


@dataclass(frozen=True, order=True, slots=True)
class Money:
    micros: int

    def __post_init__(self) -> None:
        if not isinstance(self.micros, int) or isinstance(self.micros, bool):
            raise TypeError("Money.micros must be an int")

    @classmethod
    def zero(cls) -> Money:
        return cls(0)

    @classmethod
    def from_usd(cls, usd: Decimal | str | int) -> Money:
        """Convert dollars to micros, rounding *up* so a cost is never under-counted."""
        amount = Decimal(usd) * _MICROS_PER_USD
        return cls(int(amount.to_integral_value(rounding=ROUND_CEILING)))

    def to_usd(self) -> Decimal:
        return Decimal(self.micros) / _MICROS_PER_USD

    def __add__(self, other: Money) -> Money:
        return Money(self.micros + other.micros)

    def __sub__(self, other: Money) -> Money:
        return Money(self.micros - other.micros)

    def __mul__(self, factor: int) -> Money:
        return Money(self.micros * factor)

    def __str__(self) -> str:
        return f"${self.to_usd():.6f}"
