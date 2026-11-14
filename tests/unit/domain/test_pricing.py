from decimal import Decimal

import pytest

from assayer.domain.errors import ConfigError
from assayer.domain.messages import Usage
from assayer.domain.money import Money
from assayer.domain.pricing import (
    ModelPrice,
    PriceEntry,
    PriceTable,
    prompt_token_upper_bound,
)
from tests.support.fakes import a_request

PRICE = ModelPrice(input_per_mtok=Decimal("1.00"), output_per_mtok=Decimal("2.00"))


def test_cost_is_per_million_tokens_rounded_up() -> None:
    assert PRICE.cost(Usage(prompt_tokens=1_000_000, completion_tokens=0)) == Money.from_usd("1")
    assert PRICE.cost(Usage(prompt_tokens=1, completion_tokens=0)) == Money(1)


def test_worst_case_bounds_any_realistic_usage() -> None:
    request = a_request(text="naïve café " * 50, max_tokens=200)
    bound = prompt_token_upper_bound(request)
    assert bound >= len(("naïve café " * 50).encode())
    worst = PRICE.worst_case(request)
    # Even if every byte were one token and the model used all max_tokens:
    assert worst >= PRICE.cost(Usage(prompt_tokens=bound, completion_tokens=200))


def test_worst_case_requires_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        PRICE.worst_case(a_request(max_tokens=None))


def test_lookup_first_glob_wins_and_unknown_fails_closed() -> None:
    free = ModelPrice(input_per_mtok=Decimal(0), output_per_mtok=Decimal(0))
    table = PriceTable(
        entries=[
            PriceEntry(pattern="ollama/*", price=free),
            PriceEntry(pattern="*", price=PRICE),
        ]
    )
    assert table.lookup("ollama/llama3.2:3b") == free
    assert table.lookup("groq/x") == PRICE
    with pytest.raises(ConfigError, match="no price"):
        PriceTable().lookup("mystery")
    assert PriceTable().with_fallback(free).lookup("mystery") == free
