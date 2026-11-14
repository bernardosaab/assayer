"""The provider contract: behaviour every ``ModelProvider`` adapter must share.

A test module subclasses :class:`ProviderContract` as ``Test...`` and implements
:meth:`make_provider`, returning a provider primed for one named scenario. The fake, the
HTTP adapter against recorded payloads and the live Ollama run all execute the same tests, so
the fake cannot drift from what real providers actually return.

Scenarios:

* ``text`` -- a plain reply, ``"Hello there"``.
* ``tool_call`` -- one call to ``get_pto_balance`` with ``{"employee_id": "e1"}``.
* ``json`` -- with ``response_schema`` set, content parses as ``{"answer": 42}``.
* ``no_usage`` -- the provider may omit usage; it is then ``None``, never garbage.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from assayer.domain.messages import (
    CallContext,
    CompletionRequest,
    CompletionResponse,
    Message,
    Purpose,
    ToolSpec,
    Usage,
)
from assayer.ports.provider import ModelProvider

PTO_TOOL = ToolSpec(
    name="get_pto_balance",
    description="Look up an employee's remaining paid-time-off balance in days.",
    parameters={
        "type": "object",
        "properties": {"employee_id": {"type": "string", "description": "The employee id."}},
        "required": ["employee_id"],
    },
)

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def contract_request(scenario: str, model: str) -> CompletionRequest:
    """The request each scenario sends. Prompts are phrased so a small real model complies."""
    ctx = CallContext(purpose=Purpose.AGENT, case_id=f"contract-{scenario}", turn=1)
    if scenario in ("text", "no_usage"):
        return CompletionRequest(
            model=model,
            context=ctx,
            messages=(
                Message.system("You are a terse assistant. Reply with exactly: Hello there"),
                Message.user("Greet me."),
            ),
            max_tokens=32,
            seed=7,
        )
    if scenario == "tool_call":
        return CompletionRequest(
            model=model,
            context=ctx,
            messages=(
                Message.system("You are an HR assistant. Always use tools to look up data."),
                Message.user("How many PTO days does employee e1 have left?"),
            ),
            tools=(PTO_TOOL,),
            max_tokens=128,
            seed=7,
        )
    if scenario == "json":
        return CompletionRequest(
            model=model,
            context=ctx,
            messages=(
                Message.system('Answer in JSON of the form {"answer": <integer>}.'),
                Message.user("What is 6 times 7?"),
            ),
            response_schema=ANSWER_SCHEMA,
            max_tokens=64,
            seed=7,
        )
    raise ValueError(f"unknown contract scenario {scenario!r}")


class ProviderContract:
    """Mixin: shared contract tests. Subclasses implement :meth:`make_provider`."""

    #: Model id sent in contract requests (namespaced like the harness does it).
    model: ClassVar[str] = "fake/contract-model"
    #: Exact-content checks. A live model is only held to shapes, not exact wording.
    strict_content: ClassVar[bool] = True

    async def make_provider(self, scenario: str) -> ModelProvider:
        raise NotImplementedError

    async def dispose(self, provider: ModelProvider) -> None:
        """Release the provider's resources. Override for providers holding connections."""

    async def run_scenario(self, scenario: str) -> tuple[CompletionRequest, CompletionResponse]:
        provider = await self.make_provider(scenario)
        try:
            request = contract_request(scenario, self.model)
            return request, await provider.complete(request)
        finally:
            await self.dispose(provider)

    @staticmethod
    def assert_common(request: CompletionRequest, response: CompletionResponse) -> None:
        assert isinstance(response, CompletionResponse)
        assert response.message.role == "assistant"
        assert response.model == request.model
        assert response.message.content is None or isinstance(response.message.content, str)
        assert response.usage is None or isinstance(response.usage, Usage)
        assert response.latency_ms >= 0.0

    async def test_text_reply(self) -> None:
        request, response = await self.run_scenario("text")
        self.assert_common(request, response)
        assert isinstance(response.message.content, str)
        assert response.message.content.strip()
        assert response.message.tool_calls == ()
        if self.strict_content:
            assert response.message.content == "Hello there"

    async def test_tool_call(self) -> None:
        request, response = await self.run_scenario("tool_call")
        self.assert_common(request, response)
        assert len(response.message.tool_calls) >= 1
        call = response.message.tool_calls[0]
        assert isinstance(call.id, str)
        assert call.id
        assert call.name == "get_pto_balance"
        assert isinstance(call.arguments, dict)
        if self.strict_content:
            assert len(response.message.tool_calls) == 1
            assert call.arguments == {"employee_id": "e1"}

    async def test_json_response(self) -> None:
        request, response = await self.run_scenario("json")
        self.assert_common(request, response)
        assert response.message.content is not None
        parsed = json.loads(response.message.content)
        assert isinstance(parsed, dict)
        assert "answer" in parsed
        if self.strict_content:
            assert parsed == {"answer": 42}

    async def test_usage_none_or_valid(self) -> None:
        request, response = await self.run_scenario("no_usage")
        self.assert_common(request, response)
        if response.usage is not None:
            assert response.usage.prompt_tokens >= 0
            assert response.usage.completion_tokens >= 0
