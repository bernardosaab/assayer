"""Property tests for verbatim-quote verification (ADR-0011)."""

from hypothesis import assume, given
from hypothesis import strategies as st

from assayer.domain.quotes import normalize, verify_quote

words = st.text(alphabet=st.characters(categories=["Ll", "Lu", "Nd"]), min_size=2, max_size=8)
sentences = st.lists(words, min_size=4, max_size=30).map(" ".join)
ascii_words = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    min_size=2,
    max_size=8,
)
ascii_sentences = st.lists(ascii_words, min_size=4, max_size=30).map(" ".join)


@given(st.text())
def test_normalize_is_idempotent(text: str) -> None:
    assert normalize(normalize(text)) == normalize(text)


def _draw_substring(source: str, data: st.DataObject) -> str:
    tokens = source.split(" ")
    start = data.draw(st.integers(0, len(tokens) - 1))
    end = data.draw(st.integers(start + 1, len(tokens)))
    return " ".join(tokens[start:end])


@given(sentences, st.data())
def test_substring_verifies_despite_whitespace_and_wrapping_quotes(
    source: str, data: st.DataObject
) -> None:
    quote = _draw_substring(source, data)
    assume(len(normalize(quote)) >= 8)
    perturbed = chr(0x201C) + quote.replace(" ", "  \n ") + chr(0x201D)
    assert verify_quote(perturbed, source)


@given(ascii_sentences, st.data())
def test_substring_verifies_despite_case(source: str, data: st.DataObject) -> None:
    # Case is only perturbed for ASCII: in full Unicode, upper() is not invertible by casefold()
    # (e.g. dotless i: "ı".upper() == "I", and "I".casefold() == "i"). Found by Hypothesis.
    quote = _draw_substring(source, data)
    assume(len(quote) >= 8)
    assert verify_quote(quote.swapcase(), source)


@given(sentences, words)
def test_quote_with_foreign_token_never_verifies(source: str, extra: str) -> None:
    marker = "zzq" + extra + "qzz"
    assume(normalize(marker) not in normalize(source))
    first = source.split(" ", maxsplit=1)[0]
    assert not verify_quote(f"{first} {marker}", source)


def test_short_quotes_are_rejected() -> None:
    assert not verify_quote("the", "the cat sat on the mat")
    assert verify_quote("cat sat on", "the cat sat on the mat")


def test_typographic_variants() -> None:
    source = "He said “it’s fine” — then left…"
    assert verify_quote('"it\'s fine" - then left', source)
    assert verify_quote("...it's fine\" - then left...", source)
