"""OpenAI-compatible HTTP adapter: contract on recorded payloads, plus mapping and errors.

Every test drives the real adapter through ``httpx.MockTransport``, so request shaping, status
handling and response parsing are exercised end to end without a network.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from assayer.adapters.providers.openai_compat import OpenAICompatProvider
from assayer.domain.errors import (
    ProviderBadRequest,
    ProviderError,
    ProviderInvalidResponse,
    ProviderRateLimited,
    ProviderServerError,
    ProviderTimeout,
    ProviderUnavailable,
)
from assayer.domain.messages import (
    CallContext,
    CompletionRequest,
    Message,
    Purpose,
    ToolCall,
)
from assayer.ports.provider import ModelProvider
from tests.contract.provider_contract import (
    ANSWER_SCHEMA,
    PTO_TOOL,
    ProviderContract,
)

RECORDED = Path(__file__).resolve().parents[1] / "data" / "recorded_http"
BASE_URL = "http://ollama.test/v1"
MODEL = "ollama/llama3.2:3b"
SECRET = "sk-test-secret-123"

Handler = Callable[[httpx.Request], httpx.Response]


def recorded(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((RECORDED / f"{name}.json").read_text(encoding="utf-8"))
    return data


def make(handler: Handler, *, api_key: str | None = None) -> OpenAICompatProvider:
    return OpenAICompatProvider("ollama", BASE_URL, api_key, transport=httpx.MockTransport(handler))


def request_for(**overrides: Any) -> CompletionRequest:
    fields: dict[str, Any] = {
        "model": MODEL,
        "messages": (Message.user("hi"),),
        "context": CallContext(purpose=Purpose.AGENT),
    }
    fields.update(overrides)
    return CompletionRequest(**fields)


class Recorder:
    """A MockTransport handler that captures requests and replays canned responses in order."""

    def __init__(self, *responses: httpx.Response) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]

    def body(self, index: int = -1) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.requests[index].content)
        return data


@pytest.fixture
async def ok_text() -> AsyncIterator[tuple[OpenAICompatProvider, Recorder]]:
    rec = Recorder(httpx.Response(200, json=recorded("text")))
    async with make(rec) as provider:
        yield provider, rec


# ------------------------------------------------------------------ contract


class TestOpenAICompatContract(ProviderContract):
    model = MODEL

    async def make_provider(self, scenario: str) -> ModelProvider:
        payload = recorded(scenario)
        return make(lambda _req: httpx.Response(200, json=payload))

    async def dispose(self, provider: ModelProvider) -> None:
        assert isinstance(provider, OpenAICompatProvider)
        await provider.aclose()


# ------------------------------------------------------------------ request mapping


async def test_request_body_shape(ok_text: tuple[OpenAICompatProvider, Recorder]) -> None:
    provider, rec = ok_text
    request = request_for(
        messages=(
            Message.system("be nice"),
            Message.user("pto for e1?"),
            Message.assistant(
                None,
                (ToolCall(id="call_1", name="get_pto_balance", arguments={"employee_id": "e1"}),),
            ),
            Message.tool("call_1", '{"days": 12}'),
        ),
        tools=(PTO_TOOL,),
        response_schema=ANSWER_SCHEMA,
        max_tokens=100,
        temperature=0.5,
        seed=3,
    )

    response = await provider.complete(request)

    sent = rec.requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == f"{BASE_URL}/chat/completions"
    body = rec.body()
    assert body["model"] == "llama3.2:3b"
    assert response.model == MODEL
    assert body["max_tokens"] == 100
    assert body["temperature"] == 0.5
    assert body["seed"] == 3
    assert body["messages"] == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "pto for e1?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_pto_balance", "arguments": '{"employee_id": "e1"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"days": 12}'},
    ]
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_pto_balance",
                "description": PTO_TOOL.description,
                "parameters": PTO_TOOL.parameters,
            },
        }
    ]
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": ANSWER_SCHEMA},
    }


async def test_optional_fields_omitted(ok_text: tuple[OpenAICompatProvider, Recorder]) -> None:
    provider, rec = ok_text
    await provider.complete(request_for())
    body = rec.body()
    for key in ("max_tokens", "seed", "tools", "response_format"):
        assert key not in body
    assert body["temperature"] == 0.0


async def test_prefix_only_stripped_for_own_namespace() -> None:
    rec = Recorder(httpx.Response(200, json=recorded("text")))
    async with make(rec) as provider:
        response = await provider.complete(request_for(model="groq/llama-3.1-8b"))
    assert rec.body()["model"] == "groq/llama-3.1-8b"
    assert response.model == "groq/llama-3.1-8b"


async def test_api_key_sent_as_bearer_when_set() -> None:
    rec = Recorder(httpx.Response(200, json=recorded("text")))
    async with make(rec, api_key=SECRET) as provider:
        await provider.complete(request_for())
    assert rec.requests[0].headers["Authorization"] == f"Bearer {SECRET}"


async def test_no_authorization_header_without_key(
    ok_text: tuple[OpenAICompatProvider, Recorder],
) -> None:
    provider, rec = ok_text
    await provider.complete(request_for())
    assert "Authorization" not in rec.requests[0].headers


async def test_extra_headers_sent() -> None:
    rec = Recorder(httpx.Response(200, json=recorded("text")))
    provider = OpenAICompatProvider(
        "openrouter",
        BASE_URL,
        transport=httpx.MockTransport(rec),
        extra_headers={"X-Title": "assayer"},
    )
    async with provider:
        await provider.complete(request_for())
    assert rec.requests[0].headers["X-Title"] == "assayer"


# ------------------------------------------------------------------ response mapping


async def test_usage_and_latency_mapped(ok_text: tuple[OpenAICompatProvider, Recorder]) -> None:
    provider, _ = ok_text
    response = await provider.complete(request_for())
    assert response.usage is not None
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (38, 3)
    assert response.latency_ms >= 0.0


async def test_invalid_tool_arguments_kept_raw() -> None:
    payload = recorded("tool_call")
    payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{employee: e1"
    async with make(lambda _r: httpx.Response(200, json=payload)) as provider:
        response = await provider.complete(request_for())
    call = response.message.tool_calls[0]
    assert call.name == "get_pto_balance"
    assert call.arguments == {"_raw": "{employee: e1"}


async def test_non_object_tool_arguments_kept_raw() -> None:
    payload = recorded("tool_call")
    payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "[1, 2]"
    async with make(lambda _r: httpx.Response(200, json=payload)) as provider:
        response = await provider.complete(request_for())
    assert response.message.tool_calls[0].arguments == {"_raw": "[1, 2]"}


async def test_empty_content_with_tool_calls_is_none() -> None:
    async with make(lambda _r: httpx.Response(200, json=recorded("tool_call"))) as provider:
        response = await provider.complete(request_for())
    assert response.message.content is None


# ------------------------------------------------------------------ error mapping


async def _raise_from(exc: Exception) -> ProviderError:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    async with make(handler, api_key=SECRET) as provider:
        with pytest.raises(ProviderError) as info:
            await provider.complete(request_for())
    assert SECRET not in str(info.value)
    return info.value


async def _status(status: int, **kwargs: Any) -> ProviderError:
    async with make(lambda _r: httpx.Response(status, **kwargs), api_key=SECRET) as provider:
        with pytest.raises(ProviderError) as info:
            await provider.complete(request_for())
    assert SECRET not in str(info.value)
    return info.value


async def test_connect_error_is_unavailable() -> None:
    err = await _raise_from(httpx.ConnectError("connection refused"))
    assert isinstance(err, ProviderUnavailable)
    assert not err.request_sent
    assert err.retryable


async def test_connect_timeout_is_unavailable() -> None:
    err = await _raise_from(httpx.ConnectTimeout("timed out connecting"))
    assert isinstance(err, ProviderUnavailable)


async def test_read_timeout_is_timeout() -> None:
    err = await _raise_from(httpx.ReadTimeout("read timed out"))
    assert isinstance(err, ProviderTimeout)
    assert err.request_sent


async def test_read_error_after_send_is_server_error() -> None:
    err = await _raise_from(httpx.RemoteProtocolError("peer closed connection"))
    assert isinstance(err, ProviderServerError)
    assert err.request_sent
    assert err.retryable


async def test_429_with_retry_after() -> None:
    err = await _status(429, headers={"Retry-After": "7"}, json={"error": "slow down"})
    assert isinstance(err, ProviderRateLimited)
    assert err.retry_after_s == 7.0
    assert "429" in str(err)


async def test_429_with_http_date_retry_after_is_none() -> None:
    err = await _status(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
    assert isinstance(err, ProviderRateLimited)
    assert err.retry_after_s is None


async def test_500_is_server_error() -> None:
    err = await _status(500, text="llama runner process has terminated")
    assert isinstance(err, ProviderServerError)
    assert "500" in str(err)
    assert "llama runner" in str(err)


async def test_404_hints_model_pull() -> None:
    err = await _status(404, json={"error": {"message": 'model "llama3.2:3b" not found'}})
    assert isinstance(err, ProviderBadRequest)
    assert not err.retryable
    assert "404" in str(err)
    assert "ollama pull llama3.2:3b" in str(err)


async def test_401_is_bad_request_mentioning_key() -> None:
    err = await _status(401, json={"error": "invalid api key"})
    assert isinstance(err, ProviderBadRequest)
    assert "401" in str(err)
    assert "API key" in str(err)


async def test_long_error_body_truncated() -> None:
    err = await _status(400, text="x" * 5000)
    assert isinstance(err, ProviderBadRequest)
    assert len(str(err)) < 1000


async def test_invalid_json_body() -> None:
    err = await _status(200, text="<html>gateway</html>")
    assert isinstance(err, ProviderInvalidResponse)


async def test_missing_choices() -> None:
    err = await _status(200, json={"id": "x", "object": "chat.completion"})
    assert isinstance(err, ProviderInvalidResponse)


async def test_empty_choices() -> None:
    err = await _status(200, json={"choices": []})
    assert isinstance(err, ProviderInvalidResponse)


# ------------------------------------------------------------------ json_schema fallback


async def test_json_schema_falls_back_to_json_object(caplog: pytest.LogCaptureFixture) -> None:
    rejection = httpx.Response(
        400, json={"error": {"message": "response_format.type 'json_schema' is not supported"}}
    )
    rec = Recorder(rejection, httpx.Response(200, json=recorded("json")))
    request = request_for(response_schema=ANSWER_SCHEMA)

    with caplog.at_level(logging.INFO, logger="assayer.adapters.providers.openai_compat"):
        async with make(rec) as provider:
            first = await provider.complete(request)
            second = await provider.complete(request)

    assert json.loads(first.message.content or "") == {"answer": 42}
    assert json.loads(second.message.content or "") == {"answer": 42}
    formats = [rec.body(i)["response_format"]["type"] for i in range(len(rec.requests))]
    assert formats == ["json_schema", "json_object", "json_object"]
    assert any("json_schema" in r.getMessage() for r in caplog.records)


async def test_unrelated_400_does_not_trigger_fallback() -> None:
    rec = Recorder(httpx.Response(400, json={"error": "context length exceeded"}))
    async with make(rec) as provider:
        with pytest.raises(ProviderBadRequest):
            await provider.complete(request_for(response_schema=ANSWER_SCHEMA))
    assert len(rec.requests) == 1


async def test_fallback_rejected_again_raises() -> None:
    rec = Recorder(httpx.Response(422, text="unsupported response_format"))
    async with make(rec) as provider:
        with pytest.raises(ProviderBadRequest):
            await provider.complete(request_for(response_schema=ANSWER_SCHEMA))
    assert len(rec.requests) == 2


# ------------------------------------------------------------------ list_models


async def test_list_models() -> None:
    rec = Recorder(httpx.Response(200, json=recorded("models")))
    async with make(rec, api_key=SECRET) as provider:
        models = await provider.list_models()
    assert models == ["llama3.2:3b", "qwen2.5:7b"]
    sent = rec.requests[0]
    assert sent.method == "GET"
    assert str(sent.url) == f"{BASE_URL}/models"
    assert sent.headers["Authorization"] == f"Bearer {SECRET}"


async def test_list_models_errors_mapped() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with make(refuse) as provider:
        with pytest.raises(ProviderUnavailable):
            await provider.list_models()
    async with make(lambda _r: httpx.Response(200, json={"object": "list"})) as provider:
        with pytest.raises(ProviderInvalidResponse):
            await provider.list_models()


async def test_name_property() -> None:
    async with make(lambda _r: httpx.Response(200)) as provider:
        assert provider.name == "ollama"
