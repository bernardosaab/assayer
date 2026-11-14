"""An eval definition: everything needed to run and grade one eval, already resolved from disk.

Version hashes are derived from content, never bumped by hand (ADR-0015): editing a prompt
changes ``prompt_version``, and editing the rubric changes ``judge_version``.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field

from assayer.domain.base import Model, content_hash
from assayer.domain.fixtures import CheckSpec, ConversationFixture, Criterion, TaskFixture

# 2: the subject of extraction cases is the parsed output (with derived fields), not raw text.
JUDGE_PROTOCOL_VERSION = "2"


class EvalKind(StrEnum):
    TASK = "task"
    CONVERSATION = "conversation"


class TaskSpec(Model):
    runner: str = "extraction"
    system_prompt: str
    output_schema: dict[str, Any] | None = None
    args: dict[str, Any] = Field(default_factory=dict)


class AgentSpec(Model):
    runner: str = "tool_agent"
    system_prompt: str
    environment: str  # "file.py:FactoryClass", relative to the eval directory
    max_tool_steps: int = Field(default=5, ge=1)
    args: dict[str, Any] = Field(default_factory=dict)


class SimulatorSpec(Model):
    system_prompt: str


class JudgeSpec(Model):
    criteria: list[Criterion]
    instructions: str = ""
    min_quote_chars: int = Field(default=8, ge=1)


class EvalSpec(Model):
    name: str
    kind: EvalKind
    description: str = ""
    root: Path
    task: TaskSpec | None = None
    agent: AgentSpec | None = None
    simulator: SimulatorSpec | None = None
    checks: list[CheckSpec] = Field(default_factory=list)
    judge: JudgeSpec | None = None
    cases: list[TaskFixture | ConversationFixture]
    fake_script: Path | None = None

    @property
    def prompt_version(self) -> str:
        """Hash of everything that defines the system under test (not the graders)."""
        return content_hash(self.task, self.agent, self.simulator)

    def judge_version(self, judge_model: str | None) -> str | None:
        if self.judge is None or judge_model is None:
            return None
        return content_hash(JUDGE_PROTOCOL_VERSION, self.judge, judge_model)

    def criteria_for(self, case: TaskFixture | ConversationFixture) -> list[Criterion]:
        """Eval-wide rubric lines plus a scenario's own judge criteria."""
        criteria = list(self.judge.criteria) if self.judge else []
        if isinstance(case, ConversationFixture):
            criteria.extend(case.scenario.expectations.judge_criteria)
        return criteria

    def checks_for(self, case: TaskFixture | ConversationFixture) -> list[CheckSpec]:
        """Eval-wide checks plus the case's own (fixture or scenario expectations)."""
        own = case.checks if isinstance(case, TaskFixture) else case.scenario.expectations.checks
        return [*self.checks, *own]
