# 0020. Static Jinja2 HTML reports, published to GitHub Pages

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

Error analysis means reading attempts: the output or transcript, which checks failed, and what
the judge quoted. Terminal tables are too dense for that, and a dashboard service is something
else to host, secure and keep alive.

## Considered options

1. A web dashboard (Streamlit, FastAPI + frontend).
2. A notebook.
3. A static, self-contained HTML file per run and per comparison.

## Decision

Option 3: `assayer report RUN` and `assayer diff BASE CAND --html`.

- `application/report.py` builds a view model from the store (analytics, per-attempt checks,
  judge reasoning and findings with verified/dropped marks, outputs).
- `cli/html.py` renders it with Jinja2 templates in `cli/templates/`: inline CSS with light and
  dark themes, no JavaScript, native `<details>` for disclosure, autoescaping on.
- Rendering application view models is presentation, like the terminal renderer, so it lives in
  the outermost layer (`cli`), not among the adapters. The layering test caught the first
  placement under `adapters/`.
- `scripts/build_site.py` runs every eval offline and the Pages workflow publishes the reports.
- Reports are golden-tested byte for byte (`tests/golden/`), with a fixed clock and fake git
  metadata; this caught a hash-randomized ordering of failure causes.

## Consequences

**Positive:** a report opens anywhere, can be attached to a pull request or an issue, works
offline, and is easy to test. The demo site costs nothing to host.

**Negative / trade-offs:** no filtering, sorting or drill-down beyond what the template shows;
large runs make large files. Any template change requires regenerating the golden files
(`pytest --update-golden`) and reviewing the diff.
