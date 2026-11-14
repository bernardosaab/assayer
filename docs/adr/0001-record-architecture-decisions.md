# 0001. Record architecture decisions

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

An eval harness is full of decisions that look arbitrary in code but were made for a reason:
why a crashed attempt is not retried automatically, why the cost bound over-reserves, why a
judge verdict is overridden by its own findings. Without a written record, those reasons are
lost and the decisions get "fixed" back into bugs.

## Considered options

1. Comments in code.
2. A long design document.
3. Lightweight Architecture Decision Records (MADR format), one file per decision.

## Decision

Option 3. Every decision that a reasonable engineer could make differently gets an ADR in
`docs/adr/`, numbered, immutable once accepted (superseded rather than edited).

## Consequences

**Positive:** reviewers can read the *why* next to the *what*; trade-offs are explicit.

**Negative / trade-offs:** some writing overhead per decision; ADRs can drift from code if not
updated in the same PR, so the PR template asks for it.
