"""The provider contract against a real local Ollama. Opt in with ``pytest -m live``.

Deselected by default (``-m "not live"``) and skipped when Ollama is not reachable, so it never
breaks CI. The model comes from ``ASSAYER_LIVE_MODEL`` (default ``ollama/llama3.2:3b``). A real
model is held to response *shapes* only, not exact wording.
"""

from __future__ import annotations

import os

import pytest

from assayer.adapters.providers.openai_compat import OpenAICompatProvider
from assayer.domain.errors import ProviderError
from assayer.ports.provider import ModelProvider
from tests.contract.provider_contract import ProviderContract

pytestmark = pytest.mark.live

OLLAMA_URL = os.environ.get("ASSAYER_LIVE_OLLAMA_URL", "http://localhost:11434/v1")
LIVE_MODEL = os.environ.get("ASSAYER_LIVE_MODEL", "ollama/llama3.2:3b")


def _provider() -> OpenAICompatProvider:
    return OpenAICompatProvider("ollama", OLLAMA_URL, timeout_s=300)


# Probe once per session: a refused connection can take seconds, don't pay it per test.
_skip_reason: list[str | None] = []


async def _probe() -> str | None:
    async with OpenAICompatProvider("ollama", OLLAMA_URL, timeout_s=5) as provider:
        try:
            models = await provider.list_models()
        except ProviderError as exc:
            return f"Ollama not reachable at {OLLAMA_URL}: {exc}"
    wanted = LIVE_MODEL.removeprefix("ollama/")
    if wanted not in models:
        return f"model {wanted!r} not pulled (run `ollama pull {wanted}`)"
    return None


@pytest.fixture(autouse=True)
async def _require_ollama() -> None:
    if not _skip_reason:
        _skip_reason.append(await _probe())
    if _skip_reason[0] is not None:
        pytest.skip(_skip_reason[0])


class TestLiveOllamaContract(ProviderContract):
    model = LIVE_MODEL
    strict_content = False

    async def make_provider(self, scenario: str) -> ModelProvider:
        return _provider()

    async def dispose(self, provider: ModelProvider) -> None:
        assert isinstance(provider, OpenAICompatProvider)
        await provider.aclose()


async def test_live_list_models_includes_configured_model() -> None:
    async with _provider() as provider:
        models = await provider.list_models()
    assert LIVE_MODEL.removeprefix("ollama/") in models
