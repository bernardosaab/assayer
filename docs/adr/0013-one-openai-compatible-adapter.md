# 0013. One OpenAI-compatible httpx adapter instead of vendor SDKs

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

The harness must run free (local Ollama), on hosted free tiers (Groq, OpenRouter) and on
self-hosted servers (vLLM). Every one of them speaks the OpenAI `/chat/completions` wire format.
Each vendor SDK adds a dependency, its own retry logic (which would double-count against our
cost guard) and its own exception types.

## Considered options

1. One SDK per vendor behind the `ModelProvider` port.
2. A multi-vendor wrapper library.
3. One adapter over `httpx` for the OpenAI-compatible protocol.

## Decision

Option 3: `OpenAICompatProvider` in `adapters/providers/openai_compat.py`.

- **Model namespacing.** Harness model ids carry a prefix for pricing (`ollama/llama3.2:3b`).
  The adapter strips only its own `name/` prefix before sending, and keeps the full id in the
  response, so the price table and the ledger see one stable name.
- **Structured output with a fallback.** `response_schema` is sent as
  `response_format: {"type": "json_schema", ...}`. If the server answers 400/422 and the body
  mentions `response_format`/`json_schema`, the call is retried once with `json_object`, and the
  instance remembers that `json_schema` is unsupported.
- **Errors carry billing semantics** (`domain/errors.py`): connection failures and pool timeouts
  become `ProviderUnavailable` (`request_sent=False`, so the reservation is released); read
  timeouts, protocol errors and 5xx become retryable errors with `request_sent=True` (the call
  may have been billed); 429 becomes `ProviderRateLimited` with a numeric `Retry-After`; other
  4xx become `ProviderBadRequest` with a hint (for a 404: "is the model pulled?"). Bodies are
  truncated in messages and the API key never appears.
- **Agent misbehaviour is not an adapter error.** Tool-call arguments that are not a JSON object
  are kept as `{"_raw": ...}` so the agent is graded `fail`, not recorded as a harness error.
- **No retries inside the adapter.** Retries live in `RetryingProvider` (ADR-0019), outside the
  meter, so every retry is reserved and settled on its own.

The adapter passes the same `ProviderContract` suite as the scripted fake
(`tests/contract/`), replaying recorded Ollama-style payloads through `httpx.MockTransport`, and
the same suite runs against a live Ollama when one is available (`-m live`).

## Consequences

**Positive:** one small, fully tested adapter covers every free option; no vendor SDKs; error
semantics feed the cost guard directly.

**Negative / trade-offs:** vendor-specific features are out of reach (Anthropic prompt caching,
Gemini's native long-context options, provider-side batching). Adding one means a second
adapter behind the same port and the same contract tests. Servers that claim OpenAI
compatibility differ in edge cases (usage reporting, tool-call ids); those surface as contract
test failures rather than being hidden by an SDK.
