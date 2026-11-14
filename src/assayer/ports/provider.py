"""The model-provider port. Every model call in the harness goes through this one method."""

from __future__ import annotations

from typing import Protocol

from assayer.domain.messages import CompletionRequest, CompletionResponse


class ModelProvider(Protocol):
    @property
    def name(self) -> str: ...

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Run one chat completion.

        Raises:
            ProviderError: a subclass describing what went wrong and whether it was billed.
        """
        ...
