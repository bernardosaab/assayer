# Architecture Decision Records

Each record states the context, the options considered, the decision, and its consequences,
including the downsides. New records copy [`template.md`](template.md).

| # | Decision | Status |
|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions as ADRs | accepted |
| [0002](0002-hexagonal-architecture.md) | Hexagonal architecture with a pure domain, enforced by a test | accepted |
| [0003](0003-sqlite-single-results-store.md) | SQLite as the single results store, WAL, one writer thread | accepted |
| [0004](0004-attempt-unit-of-durability.md) | The attempt is the unit of durability, claimed before the first call | accepted |
| [0005](0005-interrupted-after-charge.md) | Interrupted attempts that spent money are surfaced, not silently retried | accepted |
| [0006](0006-reserve-then-settle-cost-guard.md) | Reserve-then-settle cost guard with a bytes-as-tokens bound | accepted |
| [0007](0007-integer-micro-dollars-fail-closed-pricing.md) | Integer micro-dollars; unpriced models fail closed | accepted |
| [0008](0008-three-outcomes-harness-errors-excluded.md) | Three outcomes; harness errors excluded from every rate | accepted |
| [0009](0009-idempotent-scoring-stages.md) | Idempotent scoring stages keyed by content-hashed versions | accepted |
| [0010](0010-judge-protocol.md) | Judge protocol: reasoning first, re-ask on contradiction, verdict from verified findings | accepted |
| [0011](0011-verbatim-quote-verification.md) | Verbatim quote verification and its normalization rules | accepted |
| [0012](0012-judge-from-different-model-family.md) | Judge from a different model family than the agent | accepted |
| [0013](0013-one-openai-compatible-adapter.md) | One OpenAI-compatible httpx adapter instead of vendor SDKs | accepted |
| [0014](0014-scripted-fake-provider.md) | Scripted fake provider and a network guard: CI never calls a model | accepted |
| [0015](0015-yaml-evals-and-content-hashed-versions.md) | YAML eval definitions; versions are content hashes | accepted |
| [0016](0016-gold-labels-in-yaml.md) | Gold labels with votes, tiebreaks, disputes and traps | accepted |
| [0017](0017-replay-semantics.md) | Replay re-scores stored outputs with zero agent calls | accepted |
| [0018](0018-paired-statistics-and-release-gate.md) | Wilson intervals, pass^k, paired McNemar, release gate | accepted |
| [0019](0019-asyncio-concurrency-and-middleware-order.md) | asyncio concurrency, middleware order, circuit breaker | accepted |
| [0020](0020-static-html-reports.md) | Static Jinja2 HTML reports on GitHub Pages | accepted |
