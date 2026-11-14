# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses semantic versioning.

## [0.2.1] - 2026-11-14

Two code reviews of the whole codebase, then of the fixes themselves. Every confirmed finding is
fixed in the code, with a regression test that reproduces the original failure.

### Fixed

- Reports, summaries and diffs now read one consistent grading view (grades, the judge version
  that produced them, and only the check results they used), keyed by the run's stored cases.
- `diff` warns when two runs were graded with different graders and names the stale side.
- The judge grades the parsed output, including fields derived in code, like the checks do.
- Timelines pair tool calls with their results by matching, not by id alone (some providers repeat
  or omit ids); turn numbers match the live trace. Fallback tool-call ids are unique.
- Environments without `briefing()` keep working.
- Bad `derive` rules and runner arguments are clean configuration errors, caught by `validate`.
- Live view: attempts released by a budget halt are not shown as errors; the judge progress bar
  counts only the work that will run; skipped judgements are labelled as skipped.
- Removed the `--round` option, which could never succeed (a run id is unique); the error for a
  changed configuration now says what to do instead.
- HR environment: date ranges are capped; overlapping requests (pending or approved) are rejected;
  unquoted YAML dates are accepted; ids continue after seeded gaps; policy lookup matches whole
  words only; a fixture with an unknown current user fails at creation; the balance after approval
  accounts for every pending request.

## [0.2.0] - 2026-11-14

### Added

- Live terminal view of runs (`--live/--no-live`, `--trace` for turn-by-turn conversations), built on
  a run-event observer port (ADR-0021).
- Conversation timelines and committed writes in HTML reports; a "How it works" page on the demo site.
- Real local-model results (Ollama: llama3.2:3b agent, qwen2.5:7b judge) published with paired diffs.
- `Environment.briefing()` for context assembly; `derive` for numbers computed in code;
  `quote_from: subject` for judge criteria (ADR-0022).

### Fixed

- `diff`, `summary`, `failures` and `report` read grades under the version a run was graded with when
  the current graders have not been applied to it.
- The cost bound counts real UTF-8 bytes rather than JSON escape sequences (less over-reservation).
- HR environment: a repeated PTO submit is idempotent.

## [0.1.0] - 2026-11-13

### Added

- Durable runs on SQLite: attempts claimed before their first model call, resumable by run id, recovery
  of interrupted attempts (`interrupted_after_charge` when money was already spent).
- Reserve-then-settle cost guard with a sound worst-case bound, a per-run budget, and a cost ledger.
- Provider middleware: retries with jitter and `Retry-After`, per-endpoint concurrency limits, timeouts,
  metering. One OpenAI-compatible HTTP adapter (Ollama, Groq, OpenRouter, vLLM).
- Single-step task evals and conversation evals with personas, scenarios, a reference tool agent and an
  LLM user simulator.
- Deterministic checks (field match, set F1, traps, grounding, trajectory, end state, false-success and
  disclosure sensors) and composite checks.
- LLM judge protocol: reasoning first, contradiction re-ask, verbatim quote verification.
- Replay, paired diff with McNemar test and a release gate, judge calibration (TPR/TNR, kappa, corrected
  pass rate), static HTML reports, GitHub Pages demo.
- Three example evals: `resume_extraction`, `posting_extraction`, `hr_assistant`.
