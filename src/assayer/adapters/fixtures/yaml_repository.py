"""Load an eval directory (``eval.yaml`` + prompts + fixtures) into an :class:`EvalSpec`.

Layout::

    evals/<name>/
      eval.yaml               # kind, prompts, schema, checks, judge rubric
      prompts/*.md
      fixtures/*.yaml         # one case per file (or a list of cases)
      personas/*.yaml         # conversation evals: referenced by id from fixtures
      scenarios/*.yaml
      fake/script.yaml        # scripted model replies for offline runs

All YAML goes through ``yaml.safe_load`` and Pydantic validation (ADR-0015). Errors name the
file they came from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from assayer.domain.errors import ConfigError
from assayer.domain.evalspec import EvalSpec
from assayer.domain.fixtures import ConversationFixture, Persona, Scenario, TaskFixture


def read_yaml(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError:
        raise ConfigError(f"file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc


def _documents(root: Path, pattern: str | None) -> list[tuple[Path, dict[str, Any]]]:
    """All mappings from files matching ``pattern`` (a file may hold one mapping or a list)."""
    if not pattern:
        return []
    found: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.glob(pattern)):
        data = read_yaml(path)
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                raise ConfigError(f"{path}: expected a mapping, got {type(item).__name__}")
            found.append((path, item))
    return found


def _by_id[M: (Persona, Scenario)](root: Path, pattern: str | None, model: type[M]) -> dict[str, M]:
    result: dict[str, M] = {}
    for path, item in _documents(root, pattern):
        try:
            obj = model.model_validate(item)
        except ValidationError as exc:
            raise ConfigError(f"{path}: invalid {model.__name__.lower()}:\n{exc}") from exc
        if obj.id in result:
            raise ConfigError(f"{path}: duplicate {model.__name__.lower()} id {obj.id!r}")
        result[obj.id] = obj
    return result


class YamlEvalRepository:
    """Implements ``FixtureRepository``."""

    def load(self, eval_dir: Path) -> EvalSpec:
        eval_dir = eval_dir.resolve()
        raw = read_yaml(eval_dir / "eval.yaml")
        if not isinstance(raw, dict):
            raise ConfigError(f"{eval_dir / 'eval.yaml'}: expected a mapping")
        data = dict(raw)
        kind = data.get("kind", "task")
        data.setdefault("name", eval_dir.name)
        data["root"] = eval_dir

        for section in ("task", "agent", "simulator"):
            if isinstance(data.get(section), dict):
                data[section] = self._resolve_prompt(eval_dir, dict(data[section]))
        if isinstance(data.get("judge"), dict):
            judge = dict(data["judge"])
            if "instructions_file" in judge:
                judge["instructions"] = self._read_text(eval_dir / judge.pop("instructions_file"))
            data["judge"] = judge

        fixtures_glob = data.pop("fixtures", "fixtures/*.yaml")
        personas_glob = data.pop("personas", None)
        scenarios_glob = data.pop("scenarios", None)
        if kind == "conversation":
            personas = _by_id(eval_dir, personas_glob or "personas/*.yaml", Persona)
            scenarios = _by_id(eval_dir, scenarios_glob or "scenarios/*.yaml", Scenario)
            data["cases"] = self._conversation_cases(eval_dir, fixtures_glob, personas, scenarios)
        else:
            data["cases"] = self._task_cases(eval_dir, fixtures_glob)

        if script := data.pop("fake_script", "fake/script.yaml"):
            path = eval_dir / script
            data["fake_script"] = path if path.exists() else None

        try:
            spec = EvalSpec.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(f"{eval_dir / 'eval.yaml'}: invalid eval definition:\n{exc}") from exc
        self._check_unique_ids(spec)
        return spec

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise ConfigError(f"file not found: {path}") from None

    def _resolve_prompt(self, root: Path, section: dict[str, Any]) -> dict[str, Any]:
        if "prompt" in section:
            section["system_prompt"] = self._read_text(root / section.pop("prompt"))
        if isinstance(section.get("output_schema"), str):
            section["output_schema"] = read_yaml(root / section["output_schema"])
        return section

    def _task_cases(self, root: Path, pattern: str) -> list[TaskFixture]:
        cases: list[TaskFixture] = []
        for path, item in _documents(root, pattern):
            item = dict(item)  # noqa: PLW2901 - shallow copy before resolving refs
            if "input_file" in item:
                item["input_doc"] = self._read_text(path.parent / item.pop("input_file"))
            try:
                cases.append(TaskFixture.model_validate(item))
            except ValidationError as exc:
                raise ConfigError(f"{path}: invalid task fixture:\n{exc}") from exc
        return cases

    def _conversation_cases(
        self,
        root: Path,
        pattern: str,
        personas: dict[str, Persona],
        scenarios: dict[str, Scenario],
    ) -> list[ConversationFixture]:
        cases: list[ConversationFixture] = []
        for path, item in _documents(root, pattern):
            item = dict(item)  # noqa: PLW2901 - shallow copy before resolving refs
            for key, table in (("persona", personas), ("scenario", scenarios)):
                ref = item.get(key)
                if isinstance(ref, str):
                    if ref not in table:
                        raise ConfigError(f"{path}: unknown {key} {ref!r}")
                    item[key] = table[ref]
            try:
                cases.append(ConversationFixture.model_validate(item))
            except ValidationError as exc:
                raise ConfigError(f"{path}: invalid conversation fixture:\n{exc}") from exc
        return cases

    @staticmethod
    def _check_unique_ids(spec: EvalSpec) -> None:
        seen: set[str] = set()
        for case in spec.cases:
            if case.id in seen:
                raise ConfigError(f"{spec.name}: duplicate case id {case.id!r}")
            seen.add(case.id)
        if not spec.cases:
            raise ConfigError(f"{spec.name}: no fixtures found")
