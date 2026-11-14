# 0022. Context assembly, derived numbers and stricter judge evidence: fixes found by real models

- Status: accepted
- Date: 2026-11-14

## Context and problem statement

The scripted provider tests the harness; it cannot surprise it. Running the example evals against a
local `llama3.2:3b` agent and `qwen2.5:7b` judge produced four failures that no scripted test had
shown. Each could be patched with a prompt line. A prompt is advice, so each fix was placed in code
where possible.

## Decision

| Observed with a real model | Fix | Where |
|---|---|---|
| The agent booked dates in the wrong year and guessed other employees' ids | The `Environment` port gains `briefing()`: identity and today's date, appended to the agent's system prompt by the conversation loop. Context assembly is the harness's job. | `ports/sut.py`, `application/conversation.py`, `evals/hr_assistant/env.py` |
| The agent re-submitted a request after a chatty user turn, creating a duplicate write | `submit_pto_request` is idempotent: an identical pending request is returned, not duplicated | `evals/hr_assistant/env.py` |
| The extractor could not do arithmetic; a clearer prompt made it copy the reference year | The model extracts a written fact (`first_job_start_year`); the runner derives `years_experience` (`derive` in `eval.yaml`) | `builtin/extraction_task.py` |
| The judge "proved" violations by quoting correct lines of the source | A criterion can require quotes from the output under evaluation (`quote_from: subject`); other quotes are dropped | `domain/fixtures.py:Criterion`, `domain/judging.py:settle_verdict` |

Each fix has a regression test, and the before/after runs are published with paired diffs
(`docs/hill-climbing.md`, demo site).

## Consequences

**Positive:** the fixes hold for any model, not just the one that exposed them, and they are
measurable: tool-level errors in the HR eval went from 8 to 0 across the targeted classes.

**Negative / trade-offs:** `briefing()` is a new method every environment must implement. `derive`
supports one operation (`years_since`) and will need more only when a real eval asks for it.
`quote_from: subject` makes some true violations unprovable when the evidence lives only in the
source; criteria opt in one by one for that reason.
