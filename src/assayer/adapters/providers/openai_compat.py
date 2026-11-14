"""One adapter for every OpenAI-compatible ``/chat/completions`` endpoint (ADR-0013).

Ollama's ``/v1``, Groq, OpenRouter, vLLM and friends all speak the same wire format, so a single
httpx client covers them; no vendor SDKs. The adapter's real job is *classification*: every
failure becomes a :class:`ProviderError` subclass whose ``request_sent``/``retryable`` flags let
the cost guard and retry middleware do the right thing. Two rules shape that:

* Infrastructure failures are harness errors. Malformed output from the *model* (e.g. tool-call
  arguments that are not JSON) is not: it is passed through (``{"_raw": ...}``) to be graded.
* Error messages carry the status and a truncated body for debugging, never the API key.
"""

from __future__ import annotations

import json
import logging
import time
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import ValidationError

from assayer.domain.errors import (
    ProviderBadRequest,
    ProviderInvalidResponse,
    ProviderRateLimited,
    ProviderServerError,
    ProviderTimeout,
    ProviderUnavailable,
)
from assayer.domain.messages import (
    CompletionRequest,
    CompletionResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)

log = logging.getLogger(__name__)

_BODY_SNIPPET = 300
_FALLBACK_MARKERS = ("response_format", "json_schema")


def _snippet(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _BODY_SNIPPET else f"{text[:_BODY_SNIPPET]}... [truncated]"


def _message_to_wire(message: Message) -> dict[str, Any]:
    wire: dict[str, Any] = {"role": message.role}
    if message.role == "tool":
        wire["tool_call_id"] = message.tool_call_id
        wire["content"] = message.content or ""
    elif message.role == "assistant" and message.tool_calls:
        wire["content"] = message.content
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    else:
        wire["content"] = message.content or ""
    return wire


def _tool_to_wire(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string. Anything unparseable is the agent's fault."""
    if isinstance(raw, dict):  # a few servers send an object instead of a string
        return raw
    text = raw if isinstance(raw, str) else json.dumps(raw)
    try:
        parsed = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        return {"_raw": text}
    return parsed if isinstance(parsed, dict) else {"_raw": text}


def _parse_tool_calls(raw_calls: Any) -> tuple[ToolCall, ...]:
    if not raw_calls:
        return ()
    calls: list[ToolCall] = []
    for index, raw in enumerate(raw_calls):
        function = raw["function"]
        # Some servers omit ids; the conversation still needs one to pair the tool result.
        call_id = raw.get("id") or f"call_{index}"
        calls.append(
            ToolCall(
                id=call_id,
                name=function["name"],
                arguments=_parse_arguments(function.get("arguments")),
            )
        )
    return tuple(calls)


def _parse_usage(raw: Any) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    try:
        return Usage(prompt_tokens=raw["prompt_tokens"], completion_tokens=raw["completion_tokens"])
    except (KeyError, ValidationError):
        return None


class OpenAICompatProvider:
    """A :class:`~assayer.ports.provider.ModelProvider` over an OpenAI-compatible HTTP API.

    ``name`` doubles as the model namespace: with ``name="ollama"`` the harness model id
    ``ollama/llama3.2:3b`` is sent as ``llama3.2:3b``. The response keeps the namespaced id so
    pricing lookups stay consistent.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout_s: float = 120,
        transport: httpx.AsyncBaseTransport | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        headers = {"Accept": "application/json", **(extra_headers or {})}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._name = name
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers=headers,
            timeout=timeout_s,
            transport=transport,
        )
        # Flipped off the first time the server rejects json_schema; later calls skip straight
        # to json_object instead of paying for a doomed request each time.
        self._json_schema_supported = True

    @property
    def name(self) -> str:
        return self._name

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ public API

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        started = time.perf_counter()
        body = self._build_body(request)
        response = await self._send("POST", "chat/completions", body)
        if self._should_fall_back(body, response):
            log.info(
                "%s rejected response_format json_schema (HTTP %d); "
                "falling back to json_object for this provider",
                self._name,
                response.status_code,
            )
            self._json_schema_supported = False
            body = self._build_body(request)
            response = await self._send("POST", "chat/completions", body)
        self._raise_for_status(response, "POST /chat/completions", body["model"])
        data = self._json(response)
        try:
            raw_message = data["choices"][0]["message"]
            content = raw_message.get("content")
            tool_calls = _parse_tool_calls(raw_message.get("tool_calls"))
            # Tool-calling replies often carry content "" -- normalise to "no text".
            message = Message.assistant(content if content or not tool_calls else None, tool_calls)
        except (KeyError, IndexError, TypeError, AttributeError, ValidationError) as exc:
            raise ProviderInvalidResponse(
                f"{self._name}: unexpected completion shape ({type(exc).__name__}: {exc}); "
                f"body: {_snippet(response.text)}"
            ) from exc
        return CompletionResponse(
            message=message,
            model=request.model,
            usage=_parse_usage(data.get("usage")),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def list_models(self) -> list[str]:
        """Model ids as the server reports them (without the harness namespace prefix)."""
        response = await self._send("GET", "models", None)
        self._raise_for_status(response, "GET /models", None)
        data = self._json(response)
        try:
            return [str(entry["id"]) for entry in data["data"]]
        except (KeyError, TypeError) as exc:
            raise ProviderInvalidResponse(
                f"{self._name}: unexpected /models shape; body: {_snippet(response.text)}"
            ) from exc

    # ------------------------------------------------------------------ request

    def _wire_model(self, model: str) -> str:
        return model.removeprefix(f"{self._name}/")

    def _build_body(self, request: CompletionRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._wire_model(request.model),
            "messages": [_message_to_wire(m) for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.seed is not None:
            body["seed"] = request.seed
        if request.tools:
            body["tools"] = [_tool_to_wire(t) for t in request.tools]
        if request.response_schema is not None:
            if self._json_schema_supported:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": request.response_schema},
                }
            else:
                body["response_format"] = {"type": "json_object"}
        return body

    def _should_fall_back(self, body: dict[str, Any], response: httpx.Response) -> bool:
        if response.status_code not in (400, 422):
            return False
        if body.get("response_format", {}).get("type") != "json_schema":
            return False
        text = response.text.lower()
        return any(marker in text for marker in _FALLBACK_MARKERS)

    async def _send(self, method: str, path: str, body: dict[str, Any] | None) -> httpx.Response:
        where = f"{self._name} ({self._client.base_url})"
        try:
            return await self._client.request(method, path, json=body)
        # Order matters: ConnectTimeout/PoolTimeout are TimeoutExceptions, ConnectError is a
        # TransportError. Classify "never left" cases before the broader buckets.
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            raise ProviderUnavailable(
                f"{where} is unreachable: {type(exc).__name__}: {exc}. Is the server running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"{where} timed out: {type(exc).__name__}: {exc}") from exc
        except httpx.UnsupportedProtocol as exc:
            raise ProviderBadRequest(f"{where}: bad base_url: {exc}") from exc
        except httpx.TransportError as exc:
            # Read/write/protocol errors happen once bytes are on the wire: the provider may
            # have run (and billed) the request. ProviderServerError is the retryable class
            # with request_sent=True, so the cost guard keeps the reservation.
            raise ProviderServerError(
                f"{where} connection failed mid-request: {type(exc).__name__}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ response

    def _raise_for_status(self, response: httpx.Response, what: str, model: str | None) -> None:
        status = response.status_code
        if response.is_success:
            return
        detail = f"{self._name}: HTTP {status} from {what}: {_snippet(response.text)}"
        if status == 429:
            raise ProviderRateLimited(
                detail, retry_after_s=_retry_after(response.headers.get("Retry-After"))
            )
        if status >= 500:
            raise ProviderServerError(detail)
        raise ProviderBadRequest(detail + _hint(status, model))

    def _json(self, response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError
            raise ProviderInvalidResponse(
                f"{self._name}: response is not JSON (HTTP {response.status_code}): "
                f"{_snippet(response.text)}"
            ) from exc
        if not isinstance(data, dict):
            raise ProviderInvalidResponse(
                f"{self._name}: expected a JSON object, got {type(data).__name__}"
            )
        return data


def _retry_after(value: str | None) -> float | None:
    """Seconds from a numeric ``Retry-After``; HTTP-date forms are ignored (backoff decides)."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _hint(status: int, model: str | None) -> str:
    if status in (401, 403):
        return " (check the API key for this provider)"
    if status == 404:
        if model:
            return (
                f" (is model {model!r} available on this server? For Ollama: "
                f"`ollama pull {model}`; also check base_url ends in /v1)"
            )
        return " (check base_url; OpenAI-compatible servers usually end in /v1)"
    return ""


__all__ = ["OpenAICompatProvider"]
