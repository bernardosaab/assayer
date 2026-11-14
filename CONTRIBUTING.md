# Contributing

## Setup

```bash
uv sync
uv run pre-commit install
```

## Before opening a pull request

```bash
uv run ruff format . && uv run ruff check .
uv run mypy                      # strict
uv run pytest                    # no network; coverage gate 90%
```

* **Tests first.** New behaviour starts with a failing test. Domain logic gets unit or property tests;
  anything touching the store or providers gets an integration test with the scripted fake.
* **Respect the layers.** `domain` is pure; `application` depends only on `ports`; I/O lives in `adapters`;
  wiring and presentation live in `cli`. `tests/architecture/test_layering.py` enforces this.
* **Record decisions.** If a reasonable engineer could have chosen differently, add an ADR in `docs/adr/`
  (copy `template.md`) in the same pull request.
* **Golden reports.** If you change a template on purpose, regenerate with `uv run pytest --update-golden`
  and review the HTML diff.
* **No real model calls in tests.** Use `ScriptedProvider`; mark Ollama tests `@pytest.mark.live`.

## Adding an eval

See "Adding an eval" in the README. Run `uv run assayer validate <eval>` and make the eval runnable offline
with a `fake/script.yaml`, so CI can run it.
