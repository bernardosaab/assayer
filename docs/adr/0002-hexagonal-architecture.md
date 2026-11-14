# 0002. Hexagonal architecture with a pure domain, enforced by a test

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

The harness talks to many unreliable things: model endpoints (local Ollama, hosted APIs),
a database, the file system, git. The interesting logic — attempt state, cost accounting,
quote verification, grading — must be testable deterministically and offline, and adapters
must be swappable (a scripted fake model in CI, a real one locally).

## Considered options

1. A flat package where use cases call `httpx` / `sqlite3` directly.
2. Ports & adapters (hexagonal): pure `domain`, `typing.Protocol` ports, `application`
   use cases depending only on ports, `adapters` implementing them, a thin `cli` composition root.

## Decision

Option 2, with these layers and allowed dependency directions:

```
cli  ->  builtin  ->  application  ->  ports  ->  domain
           \-> adapters -------------^
```

- `domain/` is pure: no I/O, no asyncio, no frameworks beyond Pydantic.
- `ports/` contain only `Protocol`s.
- `application/` orchestrates via ports; no SQL, no HTTP.
- `adapters/` are the only place with I/O.
- `cli/app.py` builds the object graph (composition root).

The rule is enforced by `tests/architecture/test_layering.py`, a ~60-line AST scan, instead of
adding a dependency such as `import-linter`.

## Consequences

**Positive:** the domain is property-testable; every adapter has a fake; CI never needs a model.

**Negative / trade-offs:** more files and indirection than a script; `Protocol`s are structural,
so an adapter can drift from a port silently — mitigated by contract tests run against every
implementation (`tests/contract/`).
