"""Fault injection for resilience tests: scheduled provider errors and a real process crash."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping

from assayer.domain.errors import ProviderError
from assayer.domain.messages import CompletionRequest, CompletionResponse
from assayer.ports.provider import ModelProvider

CRASH_EXIT_CODE = 137


class FaultInjectingProvider:
    """Wraps a provider and misbehaves on chosen calls (1-based call numbers).

    ``hard_exit_on_call`` terminates the interpreter with ``os._exit``: no ``finally`` blocks,
    no cleanup, exactly like a ``kill -9``. It fires *after* the cost guard reserved money for
    the call, which is the scenario the crash/resume test needs.
    """

    def __init__(
        self,
        inner: ModelProvider,
        *,
        errors: Mapping[int, Callable[[], ProviderError]] | None = None,
        fail_from_call: tuple[int, Callable[[], ProviderError]] | None = None,
        hard_exit_on_call: int | None = None,
    ) -> None:
        self._inner = inner
        self._errors = dict(errors or {})
        self._fail_from = fail_from_call
        self._hard_exit_on = hard_exit_on_call
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._inner.name

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.call_count += 1
        n = self.call_count
        if self._hard_exit_on is not None and n >= self._hard_exit_on:
            os._exit(CRASH_EXIT_CODE)
        if n in self._errors:
            raise self._errors[n]()
        if self._fail_from is not None and n >= self._fail_from[0]:
            raise self._fail_from[1]()
        return await self._inner.complete(request)
