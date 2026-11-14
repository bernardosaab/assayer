# 0012. The judge comes from a different model family than the agent

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

LLM judges tend to prefer outputs from their own model family (self-preference bias). A judge
from the agent's family can inflate the pass rate, and the inflation is hardest to see exactly
when the agent improves in "its own" style. Model ids do not reliably reveal the family across
providers (`ollama/llama3.2:3b`, `groq/llama-3.1-8b-instant`, a fine-tune with a custom name).

## Considered options

1. Ignore it.
2. Infer the family from the model id and always refuse a same-family judge.
3. Declare the family per role in config, warn by default, and allow a strict mode that refuses.

## Decision

Option 3.

- `RoleProfile.family` (`config.py`) is a free string per role (`agent`, `simulator`, `judge`)
  in `assayer.toml`, defaulting to `"unknown"`. The shipped profiles pair a `llama` agent with a
  `qwen` judge (`ollama`: `llama3.2:3b` and `qwen2.5:7b`; `groq`: `llama-3.1-8b-instant` and
  `qwen/qwen3-32b`). The built-in `fake` profile uses `fake-agent` and `fake-judge`.
- `cli/container.py:check_judge_family(agent, judge, strict=...)` returns `None` in three cases:
  there is no judge, the families differ, or the agent's family is `"unknown"`. Otherwise it
  returns a warning that names the shared family and "self-preference bias is likely".
  With `strict=True` it raises `ConfigError`.
- `_assemble` calls it whenever the eval defines a judge and the profile has one, and passes the
  message to the CLI's `warn` callback. `[defaults] strict_judge_family = false` in
  `assayer.toml`. Set it to `true` in CI to turn the warning into a refusal.

A judge is a critic, so it should not be an echo of the thing it criticises. Keeping it outside
the agent's family is the cheapest structural control against shared blind spots.

## Consequences

**Positive:** the common mistake (one model for everything) is flagged the first time someone
runs a real profile. Teams that want a gate can have one through config, without code changes.

**Negative / trade-offs:**

- The family is a hand-typed label. A wrong label defeats the check, and the default `"unknown"`
  disables it *silently*, with no warning that it was skipped.
- A different family is not the same as independence. Families share training data, and a
  different-family judge brings biases of its own. This does not replace calibration against
  people (ADR-0010).
- Warning by default means it is easy to ignore in CI logs.
- Only agent vs judge is checked. The simulator's family is not, and in the `ollama` profile the
  simulator and agent are deliberately the same model.
- The judge model id is part of `judge_version`, so moving to another family re-judges every
  attempt, which costs money (ADR-0009, ADR-0017).
