# 0014. A scripted fake provider and a network guard: CI never calls a model

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

CI must be deterministic and free, and the interesting behaviour of the harness is in its
failure paths: a judge that contradicts itself, a fabricated quote, an agent that claims success
after a timeout, a provider that refuses connections, a process killed mid-call. Real models
produce these only by luck.

## Considered options

1. Record and replay real HTTP traffic (cassettes).
2. Mock the provider per test.
3. A rule-based scripted provider, plus explicit fault injection, behind the same port.

## Decision

Option 3.

- `ScriptedProvider` (`adapters/providers/fake.py`) answers from ordered rules matched on the
  call's context: purpose (agent, simulator, judge), case id, turn, the last message's role and
  content, and `nth` (counted per rule, case and attempt, so it is deterministic under
  concurrency). Tool-call ids are derived from the call's position in its conversation, not from
  global call order. A request no rule matches raises `UnscriptedCall`, so a flow change the
  script does not cover fails loudly instead of passing silently.
- Each example eval ships a `fake/script.yaml` that is **deliberately imperfect** (trap answers,
  prose instead of JSON, a contradictory judge, a fabricated quote, a false success claim), so
  offline runs exercise the real failure paths.
- `FaultInjectingProvider` (`faults.py`) raises scheduled provider errors or terminates the
  process with `os._exit` on the Nth call (the crash test, ADR-0005).
- `tests/conftest.py` blocks every non-loopback socket unless a test is marked `live`.

## Consequences

**Positive:** the whole suite, including the end-to-end crash test and the offline evals in CI,
is deterministic, fast and free. Failure paths are tested on purpose: if a control cannot fail on
camera, it cannot be trusted to stop anything.

**Negative / trade-offs:** scripts can drift from how real models behave, and they test the
harness, not model quality. Mitigations: live contract tests against Ollama (`-m live`) and a
committed real-model report on the demo site. Scripts also need updating when prompts or flows
change, which `UnscriptedCall` makes visible but not free.
