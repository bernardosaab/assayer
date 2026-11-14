"""Binding a provider to one role (agent, simulator or judge) of one attempt."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from assayer.domain.base import Model
from assayer.domain.messages import (
    CallContext,
    CompletionRequest,
    CompletionResponse,
    Message,
    ToolSpec,
)
from assayer.ports.provider import ModelProvider


class RoleSettings(Model):
    model: str
    family: str = "unknown"
    max_tokens: int = 1024
    temperature: float = 0.0
    seed: int | None = None


class BoundChatModel:
    """Implements ``ChatModel``: fills in model, limits and call context for every request."""

    def __init__(
        self, provider: ModelProvider, settings: RoleSettings, context: CallContext
    ) -> None:
        self._provider = provider
        self._settings = settings
        self._context = context
        self._calls = 0

    @property
    def model(self) -> str:
        return self._settings.model

    @property
    def calls_made(self) -> int:
        return self._calls

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        response_schema: dict[str, Any] | None = None,
        turn: int | None = None,
    ) -> CompletionResponse:
        context = self._context if turn is None else self._context.model_copy(update={"turn": turn})
        response = await self._provider.complete(
            CompletionRequest(
                model=self._settings.model,
                messages=tuple(messages),
                context=context,
                tools=tuple(tools),
                response_schema=response_schema,
                max_tokens=self._settings.max_tokens,
                temperature=self._settings.temperature,
                seed=self._settings.seed,
            )
        )
        self._calls += 1
        return response
