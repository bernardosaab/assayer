"""Load ``prices.yaml`` (ordered glob patterns, USD per million tokens)."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from assayer.adapters.fixtures.yaml_repository import read_yaml
from assayer.domain.errors import ConfigError
from assayer.domain.pricing import ModelPrice, PriceEntry, PriceTable


def load_price_table(path: Path) -> PriceTable:
    data = read_yaml(path) or []
    if not isinstance(data, list):
        raise ConfigError(
            f"{path}: expected a list of {{pattern, input_per_mtok, output_per_mtok}}"
        )
    try:
        entries = [
            PriceEntry(
                pattern=item["pattern"],
                price=ModelPrice(
                    input_per_mtok=item["input_per_mtok"], output_per_mtok=item["output_per_mtok"]
                ),
            )
            for item in data
        ]
    except (KeyError, TypeError, ValidationError) as exc:
        raise ConfigError(f"{path}: invalid price entry: {exc}") from exc
    return PriceTable(entries=entries)
