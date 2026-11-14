"""The scripted fake honours the same contract as real providers."""

from __future__ import annotations

from assayer.adapters.providers.fake import (
    Respond,
    ScriptedProvider,
    ScriptedToolCall,
    ScriptRule,
)
from assayer.domain.messages import Usage
from assayer.ports.provider import ModelProvider
from tests.contract.provider_contract import ProviderContract

_RESPONSES: dict[str, Respond] = {
    "text": Respond(content="Hello there"),
    "no_usage": Respond(content="Hello there"),
    "tool_call": Respond(
        tool_calls=[ScriptedToolCall(name="get_pto_balance", arguments={"employee_id": "e1"})]
    ),
    "json": Respond.model_validate({"json": {"answer": 42}}),
}


class TestScriptedProviderContract(ProviderContract):
    async def make_provider(self, scenario: str) -> ModelProvider:
        usage = Usage(prompt_tokens=12, completion_tokens=3) if scenario != "no_usage" else None
        return ScriptedProvider([ScriptRule(respond=_RESPONSES[scenario], usage=usage)])
