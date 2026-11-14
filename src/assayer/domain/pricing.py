"""Model prices and the worst-case cost bound (ADR-0006, ADR-0007).

The cost guard must reserve an *upper bound* on a call's cost before the call is made, so a
crash can never lose track of spend. The bound counts prompt tokens as UTF-8 bytes: byte-level
BPE tokenizers emit at most one token per byte, so this is sound for every model family without
shipping a tokenizer. It over-reserves (typically 3-4x on English), which is the trade-off.
"""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Sequence
from decimal import Decimal

from pydantic import Field

from assayer.domain.base import Model
from assayer.domain.errors import ConfigError
from assayer.domain.messages import CompletionRequest, Usage
from assayer.domain.money import Money

_PER_MESSAGE_OVERHEAD = 8
_PER_REQUEST_OVERHEAD = 64
_TOKENS_PER_MTOK = Decimal(1_000_000)


class ModelPrice(Model):
    """USD per million tokens."""

    input_per_mtok: Decimal = Field(ge=0)
    output_per_mtok: Decimal = Field(ge=0)

    def cost(self, usage: Usage) -> Money:
        usd = (
            Decimal(usage.prompt_tokens) * self.input_per_mtok
            + Decimal(usage.completion_tokens) * self.output_per_mtok
        ) / _TOKENS_PER_MTOK
        return Money.from_usd(usd)

    def worst_case(self, request: CompletionRequest) -> Money:
        if request.max_tokens is None:
            raise ValueError("worst case is unbounded without max_tokens")
        return self.cost(
            Usage(
                prompt_tokens=prompt_token_upper_bound(request),
                completion_tokens=request.max_tokens,
            )
        )


def prompt_token_upper_bound(request: CompletionRequest) -> int:
    """Sound upper bound on prompt tokens: UTF-8 bytes of everything sent, plus overheads."""
    payload = [m.model_dump(mode="json") for m in request.messages]
    tools = [t.model_dump(mode="json") for t in request.tools]
    # ensure_ascii=False counts real UTF-8 bytes rather than 6-byte escapes; still a sound bound.
    text = json.dumps([payload, tools, request.response_schema], ensure_ascii=False)
    size = len(text.encode("utf-8"))
    return size + _PER_MESSAGE_OVERHEAD * len(request.messages) + _PER_REQUEST_OVERHEAD


class PriceEntry(Model):
    pattern: str
    price: ModelPrice


class PriceTable(Model):
    """Ordered glob patterns -> prices. The first match wins; no match fails closed."""

    entries: Sequence[PriceEntry] = ()

    def lookup(self, model: str) -> ModelPrice:
        for entry in self.entries:
            if fnmatch.fnmatchcase(model, entry.pattern):
                return entry.price
        raise ConfigError(
            f"no price for model {model!r}; add it to prices.yaml "
            "(unpriced models are refused so a budget can never be bypassed)"
        )

    def with_fallback(self, price: ModelPrice) -> PriceTable:
        """Price every unknown model at ``price`` (the ``--allow-unpriced`` escape hatch)."""
        return PriceTable(entries=[*self.entries, PriceEntry(pattern="*", price=price)])
