# 0015. YAML eval definitions validated by Pydantic; versions are content hashes

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

Adding a scenario or a criterion must be cheap, reviewable in a pull request, and impossible to
get silently wrong. Every result must also say exactly which prompt, rubric and graders produced
it. Version numbers that people bump by hand are forgotten.

## Considered options

1. Evals as Python code.
2. YAML (or JSON) data with a schema, and versions bumped manually.
3. YAML data with a schema, and versions derived from content.

## Decision

Option 3.

- An eval is a directory (`eval.yaml`, `prompts/`, `fixtures/`, `personas/`, `scenarios/`,
  `fake/script.yaml`) loaded by `YamlEvalRepository` (`adapters/fixtures/yaml_repository.py`)
  with `yaml.safe_load` into Pydantic models (`domain/evalspec.py`, `domain/fixtures.py`).
  Errors name the file they came from. Conversation fixtures refer to personas and scenarios by
  id. Eval-local Python (an agent's tool environment) is loaded by file path, only from inside
  the eval directory (`plugin_loader.py`).
- `EvalSpec.prompt_version` hashes everything that defines the system under test;
  `judge_version` hashes the rubric, the judge protocol version and the judge model; check
  versions hash their arguments (ADR-0009). `domain/base.py:content_hash` uses canonical JSON.
- Every run snapshots its fixtures (`run_case.fixture_json`), so a run stays reproducible after
  the YAML changes, and a run id reused with a different configuration is refused.
- `assayer validate` (`application/validate.py`) catches unknown or duplicate checks, unresolved
  gold, traps equal to the expected value and missing sections before anything is spent.

## Consequences

**Positive:** a scenario is one file; a reviewer sees exactly what changed; nobody has to
remember to bump a version, and replay (ADR-0017) can tell grading changes from agent changes.

**Negative / trade-offs:** YAML has footguns (implicit typing, indentation); `safe_load`,
Pydantic and `validate` catch most but not all. Any edit, even whitespace in a prompt, produces a
new version, which starts a new configuration for a run id. Logic beyond the built-in runners and
checks still needs Python.
