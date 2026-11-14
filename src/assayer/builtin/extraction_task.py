"""Generic structured extraction: a document in, a JSON object out.

Any "document -> facts" eval (résumés, job postings, invoices...) can reuse this runner with only
a prompt and an output schema in YAML. That is what makes a new single-step eval cheap.

**Numbers come from code.** The model extracts facts that are written in the document; anything
that needs arithmetic is *derived* afterwards, in code (``derive`` in ``eval.yaml``). This exists
because a 3B model asked to compute years of experience copied the reference year from the
prompt instead (see ``docs/hill-climbing.md``).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, ValidationError

from assayer.application.registry import TASK_RUNNERS
from assayer.domain.base import Model
from assayer.domain.errors import ConfigError
from assayer.domain.fixtures import TaskFixture
from assayer.domain.jsontext import parse_json_object
from assayer.domain.messages import Message
from assayer.domain.outputs import TaskOutput
from assayer.ports.sut import ChatModel


class YearsSince(Model):
    """``target = as_of - <source year>``, e.g. years of experience from a first-job year."""

    years_since: str = Field(description="Extracted field holding a 4-digit year.")
    as_of: int


def _missing_required(schema: dict[str, Any] | None, value: dict[str, Any]) -> list[str]:
    if not schema:
        return []
    return [k for k in schema.get("required", []) if k not in value]


def _derive(
    parsed: dict[str, Any], rules: dict[str, YearsSince]
) -> tuple[dict[str, Any], str | None]:
    """Apply derivations. A missing or non-numeric source is an agent error, never a guess."""
    out = dict(parsed)
    for target, rule in rules.items():
        source = out.get(rule.years_since)
        try:
            year = int(str(source).strip())
        except (TypeError, ValueError):
            return out, f"cannot derive {target}: {rule.years_since}={source!r} is not a year"
        out[target] = rule.as_of - year
    return out, None


class ExtractionTask:
    def __init__(
        self,
        *,
        system_prompt: str,
        output_schema: dict[str, Any] | None = None,
        derive: dict[str, YearsSince] | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._schema = output_schema
        self._derive = derive or {}

    async def run(self, fixture: TaskFixture, llm: ChatModel) -> TaskOutput:
        response = await llm.chat(
            [Message.system(self._system_prompt), Message.user(fixture.input_doc)],
            response_schema=self._schema,
        )
        raw = response.message.content or ""
        try:
            parsed = parse_json_object(raw)
        except ValueError as exc:  # json.JSONDecodeError is a ValueError
            return TaskOutput(raw=raw, agent_error=f"output is not a JSON object: {exc}")
        missing = _missing_required(self._schema, parsed)
        if missing:
            return TaskOutput(
                raw=raw, parsed=parsed, agent_error=f"missing required fields: {missing}"
            )
        parsed, error = _derive(parsed, self._derive)
        return TaskOutput(raw=raw, parsed=parsed, agent_error=error)


@TASK_RUNNERS.register("extraction")
def _extraction(
    *,
    system_prompt: str,
    output_schema: dict[str, Any] | None = None,
    derive: dict[str, Any] | None = None,
    **unknown: Any,
) -> ExtractionTask:
    if unknown:
        raise ConfigError(f"extraction runner: unknown args {sorted(unknown)}")
    if derive is not None and not isinstance(derive, dict):
        raise ConfigError("extraction runner: derive must be a mapping of field -> rule")
    try:
        rules = {k: YearsSince.model_validate(v) for k, v in (derive or {}).items()}
    except ValidationError as exc:
        raise ConfigError(f"extraction runner: invalid derive rule:\n{exc}") from exc
    properties = (output_schema or {}).get("properties")
    if isinstance(properties, dict):
        missing = sorted(r.years_since for r in rules.values() if r.years_since not in properties)
        if missing:
            raise ConfigError(
                f"extraction runner: derive reads {missing}, "
                "which the output schema does not ask for"
            )
    return ExtractionTask(system_prompt=system_prompt, output_schema=output_schema, derive=rules)
