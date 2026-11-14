# 0011. Verbatim quote verification with narrow normalization

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

Every judge finding (ADR-0010) and every `grounded` extraction value has to be backed by text
that actually occurs in the source. The comparison must forgive typography that models routinely
change (curly quotes, dashes, case, line wrapping, a wrapping `"..."`). It must never forgive
wording, because a reworded "quote" is exactly the hallucination the check exists to catch.

## Considered options

1. Exact substring match.
2. Fuzzy matching (edit distance, token overlap, embeddings) with a threshold.
3. Substring match after a fixed, narrow normalization applied to both sides.

## Decision

Option 3, in `domain/quotes.py`.

- `normalize(text)`, applied to quote and source:
  1. Unicode NFKC.
  2. A typography table: single quotes and primes (U+2018-U+201B, U+2032) become `'`; double
     quotes (U+201C-U+201E, U+2033) become `"`; dashes and minus (U+2010-U+2015, U+2212) become
     `-`; ellipsis becomes `...`; NBSP becomes a space.
  3. `casefold()`.
  4. Collapse whitespace runs to one space, then strip.

  `normalize` is idempotent.
- `_strip_edges` (quote side only) repeatedly strips spaces, `"`, `'`, backtick, `.` and the
  ellipsis from both ends, so `"...it's fine"` matches `it's fine`.
- `verify_quote(quote, source, min_chars=8)` requires at least `min_chars` characters after
  normalization, then a plain substring test. The minimum stops "the" from verifying anything.
  It is configurable per eval as `judge.min_quote_chars`. The `grounded` check reuses
  `verify_quote` with its own `min_chars` (default 2).

`tests/property/test_quotes.py` (Hypothesis) checks the properties:

- normalization is idempotent;
- any substring still verifies after whitespace and newline perturbation and curly-quote
  wrapping;
- a quote containing a foreign token never verifies;
- short quotes are rejected.

Hypothesis found that case perturbation is safe only for ASCII. `"ı".upper()` is `"I"`, and
`"I".casefold()` is `"i"`, so a dotless i does not round-trip. The case test is therefore
restricted to ASCII sentences, and the limitation is documented there.

This is a deterministic sensor placed in front of a model critic: a hallucinated quote is caught
by code, not by asking another model.

## Consequences

**Positive:** verification is cheap, deterministic, explainable ("this text is not in the
source"), and immune to a judge that paraphrases convincingly.

**Negative / trade-offs:**

- It is strict on purpose, which costs some false negatives. A quote with a mid-sentence `...`
  elision, a fixed typo, or two non-adjacent sentences joined together is dropped, so a real
  finding is lost.
- Case-insensitivity is reliable only where `casefold` inverts `upper` (not the Turkish
  dotless i, for example).
- Eight characters is a weak floor. Common phrases verify almost anywhere.
- The match covers the whole evidence. In a transcript, a quote from the *user's* turn or a tool
  result verifies just as well as one from the agent's turn.
