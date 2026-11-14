# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses semantic versioning.

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
