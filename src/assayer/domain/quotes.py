"""Verbatim-quote verification (ADR-0011).

An LLM judge must support each violation with a quote from the source. The quote is checked
here, deterministically: a finding whose quote is not in the source is dropped, because a judge
that cannot point at the evidence is more likely hallucinating than catching a real problem.

Normalization is intentionally narrow. It forgives typography (case, curly quotes, dashes,
whitespace, wrapping quote marks, leading/trailing ellipses) but never wording.
"""

from __future__ import annotations

import re
import unicodedata

DEFAULT_MIN_CHARS = 8

# Code points rather than literal characters so the table is unambiguous to read.
_SINGLE_QUOTES = (0x2018, 0x2019, 0x201A, 0x201B, 0x2032)
_DOUBLE_QUOTES = (0x201C, 0x201D, 0x201E, 0x2033)
_DASHES = (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212)
_ELLIPSIS, _NBSP = 0x2026, 0x00A0

_TYPOGRAPHY: dict[int, str] = {
    **dict.fromkeys(_SINGLE_QUOTES, "'"),
    **dict.fromkeys(_DOUBLE_QUOTES, '"'),
    **dict.fromkeys(_DASHES, "-"),
    _ELLIPSIS: "...",
    _NBSP: " ",
}
_WS = re.compile(r"\s+")
_EDGE_NOISE = " \"'`." + chr(_ELLIPSIS)


def normalize(text: str) -> str:
    """Canonical form used on both sides of a quote comparison. Idempotent."""
    text = unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHY).casefold()
    return _WS.sub(" ", text).strip()


def _strip_edges(quote: str) -> str:
    previous = None
    while previous != quote:
        previous = quote
        quote = quote.strip(_EDGE_NOISE)
    return quote


def verify_quote(quote: str, source: str, *, min_chars: int = DEFAULT_MIN_CHARS) -> bool:
    """True iff ``quote`` occurs verbatim in ``source`` (modulo typography) and is long enough.

    The minimum length stops trivial "quotes" such as ``"the"`` from verifying anything.
    """
    needle = _strip_edges(normalize(quote))
    if len(needle) < min_chars:
        return False
    return needle in normalize(source)
