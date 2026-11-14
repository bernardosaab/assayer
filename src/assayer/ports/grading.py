"""Ports for graders: deterministic checks and the LLM judge."""

from __future__ import annotations

from typing import Protocol

from assayer.domain.checks import CheckResult
from assayer.domain.fixtures import ConversationFixture, Criterion, TaskFixture
from assayer.domain.judging import JudgeVerdict
from assayer.domain.outputs import TaskOutput, Transcript
from assayer.ports.sut import ChatModel


class Check(Protocol):
    """Strategy: one deterministic rule. ``version`` changes whenever its behaviour or args do."""

    @property
    def check_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    def evaluate(
        self, case: TaskFixture | ConversationFixture, output: TaskOutput | Transcript
    ) -> CheckResult: ...


class Judge(Protocol):
    async def judge(
        self, criteria: list[Criterion], evidence: str, subject: str, llm: ChatModel
    ) -> JudgeVerdict:
        """Grade ``subject`` against ``criteria``; quotes are verified against ``evidence``."""
        ...
