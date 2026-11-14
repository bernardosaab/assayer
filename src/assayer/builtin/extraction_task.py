"""Generic structured extraction: a document in, a JSON object out.

Any "document -> facts" eval (résumés, job postings, invoices...) can reuse this runner with only
a prompt and an output schema in YAML. That is what makes a new single-step eval cheap.
"""

from __future__ import annotations

from typing import Any

from assayer.application.registry import TASK_RUNNERS
from assayer.domain.fixtures import TaskFixture
from assayer.domain.jsontext import parse_json_object
from assayer.domain.messages import Message
from assayer.domain.outputs import TaskOutput
from assayer.ports.sut import ChatModel


def _missing_required(schema: dict[str, Any] | None, value: dict[str, Any]) -> list[str]:
    if not schema:
        return []
    return [k for k in schema.get("required", []) if k not in value]


class ExtractionTask:
    def __init__(self, *, system_prompt: str, output_schema: dict[str, Any] | None = None) -> None:
        self._system_prompt = system_prompt
        self._schema = output_schema

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
        return TaskOutput(raw=raw, parsed=parsed)


@TASK_RUNNERS.register("extraction")
def _extraction(
    *, system_prompt: str, output_schema: dict[str, Any] | None = None, **_: Any
) -> ExtractionTask:
    return ExtractionTask(system_prompt=system_prompt, output_schema=output_schema)
