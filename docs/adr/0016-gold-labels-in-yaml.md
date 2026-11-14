# 0016. Gold labels in YAML with votes, tiebreaks, disputes and traps

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

An extraction eval is only as good as its expected values. A single person's label hides
disagreement, a silent tiebreak hides a judgement call, and a plain "wrong" hides *which* mistake
the model made. Fixing the gold labels alone can move a pass rate as much as fixing the model.

## Considered options

1. A single expected value per field.
2. A labelling application with a database.
3. Structured gold in the fixture YAML, reviewed through pull requests.

## Decision

Option 3: `GoldField` in `domain/gold.py`.

- `votes`: one value per labeller (normally two). Unanimous votes are the expected value.
- `resolved`: an explicit tiebreak when votes disagree; `dispute: {assignee, note}` marks an open
  disagreement, which blocks the field until resolved.
- `traps`: plausible-but-wrong values, such as a *previous* employer for `current_employer`.
  `TrapAbsent` (`builtin/checks/fields.py`) reports a trap hit as its own failure cause.
- Shorthand `field: value` is an already-resolved label.
- `percent_agreement` and `cohens_kappa` measure labeller agreement; `assayer validate` errors on
  unresolved gold and traps equal to the expected value, and warns on single-vote fields.

`evals/resume_extraction/fixtures/` shows all of these, including a recorded tiebreak
(`tomasz-nowak`) and trap cases (`carlos-ruiz`, `mei-chen`).

## Consequences

**Positive:** disagreement and its resolution are visible and auditable in git history; a trap
hit names the mistake, which makes error analysis faster; no labelling service to build or host.

**Negative / trade-offs:** labellers edit YAML, which is fine for engineers and poor for domain
experts; there is no blind double-labelling workflow for gold (judge calibration has one,
`assayer calibrate`). Values are compared after light normalization, so a label that needs
semantic equivalence ("Sr." vs "Senior") must list the accepted form.
