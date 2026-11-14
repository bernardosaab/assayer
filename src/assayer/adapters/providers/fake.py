"""Deterministic providers for tests, CI and the offline demo (ADR-0014).

:class:`ScriptedProvider` answers from an ordered list of rules. The first rule whose ``match``
fits the request wins. A request no rule matches raises :class:`UnscriptedCall`, so a prompt or
flow change that the script does not cover fails loudly instead of passing silently.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from pydantic import ConfigDict, Field

from assayer.domain.base import Model
from assayer.domain.errors import ProviderBadRequest
from assayer.domain.messages import (
    CompletionRequest,
    CompletionResponse,
    Message,
    Purpose,
    ToolCall,
    Usage,
)


class UnscriptedCall(ProviderBadRequest):
    """The scripted provider has no rule for this request."""


class Match(Model):
    purpose: Purpose | None = None
    case_id: str | None = None  # regex, full match
    turn: int | None = None
    last_role: str | None = None
    last_content: str | None = None  # regex, searched in the last message's content
    nth: int | None = None  # 1-based: only the nth matching call within one attempt


class ScriptedToolCall(Model):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Respond(Model):
    content: str | None = None
    json_: Any = Field(default=None, alias="json")
    tool_calls: list[ScriptedToolCall] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class ScriptRule(Model):
    match: Match = Field(default_factory=Match)
    respond: Respond
    usage: Usage | None = None


def _matches(match: Match, request: CompletionRequest) -> bool:
    ctx = request.context
    last = request.messages[-1] if request.messages else None
    checks = (
        match.purpose is None or match.purpose == ctx.purpose,
        match.case_id is None
        or (ctx.case_id is not None and re.fullmatch(match.case_id, ctx.case_id) is not None),
        match.turn is None or match.turn == ctx.turn,
        match.last_role is None or (last is not None and last.role == match.last_role),
        match.last_content is None
        or (
            last is not None
            and last.content is not None
            and re.search(match.last_content, last.content, flags=re.IGNORECASE) is not None
        ),
    )
    return all(checks)


def _estimate_usage(request: CompletionRequest, text: str) -> Usage:
    prompt = sum(len(m.content or "") for m in request.messages) // 4 + 1
    return Usage(prompt_tokens=prompt, completion_tokens=len(text) // 4 + 1)


def _tool_call_id(request: CompletionRequest, index: int) -> str:
    """Stable id from the call's position in its conversation, not from global call order."""
    ctx = request.context
    seed = f"{ctx.case_id}|{ctx.attempt_id}|{ctx.turn}|{len(request.messages)}|{index}"
    return f"call_{uuid.uuid5(uuid.NAMESPACE_OID, seed).hex[:8]}"


class ScriptedProvider:
    def __init__(self, rules: Sequence[ScriptRule], *, name: str = "fake") -> None:
        self._rules = list(rules)
        self._name = name
        # nth counts per (rule, case, attempt), so it is deterministic under concurrency.
        self._hits: dict[tuple[int, str | None, int | None], int] = {}
        self.calls: list[CompletionRequest] = []

    @property
    def name(self) -> str:
        return self._name

    def calls_for(self, purpose: Purpose) -> list[CompletionRequest]:
        return [c for c in self.calls if c.context.purpose == purpose]

    def _select(self, request: CompletionRequest) -> ScriptRule:
        for index, rule in enumerate(self._rules):
            if not _matches(rule.match, request):
                continue
            key = (index, request.context.case_id, request.context.attempt_id)
            seen = self._hits.get(key, 0) + 1
            self._hits[key] = seen
            if rule.match.nth is not None and seen != rule.match.nth:
                continue
            return rule
        ctx = request.context
        last = request.messages[-1].content if request.messages else None
        raise UnscriptedCall(
            f"no script rule for purpose={ctx.purpose.value} case={ctx.case_id} "
            f"turn={ctx.turn} last={(last or '')[:80]!r}"
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        rule = self._select(request)
        respond = rule.respond
        content = respond.content
        if respond.json_ is not None:
            content = json.dumps(respond.json_)
        tool_calls = tuple(
            ToolCall(
                id=_tool_call_id(request, i),
                name=tc.name,
                arguments=tc.arguments,
            )
            for i, tc in enumerate(respond.tool_calls)
        )
        usage = rule.usage or _estimate_usage(request, content or "")
        return CompletionResponse(
            message=Message.assistant(content, tool_calls),
            model=request.model,
            usage=usage,
        )
