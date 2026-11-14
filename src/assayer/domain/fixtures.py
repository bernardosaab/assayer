"""Fixtures: one case to run.

* :class:`TaskFixture`: a single-step case (input document + gold facts).
* :class:`ConversationFixture`: a multi-turn case built from a :class:`Persona` (who the
  simulated user is) and a :class:`Scenario` (what should happen and what counts as success).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field

from assayer.domain.base import Model, content_hash
from assayer.domain.gold import GoldField


class CheckSpec(Model):
    """A reference to a deterministic check by registry name, plus its arguments."""

    use: str
    args: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None  # defaults to ``use``; set it to use one check twice

    @property
    def check_id(self) -> str:
        return self.id or self.use


class Criterion(Model):
    """One rubric line the LLM judge grades against."""

    id: str
    text: str


class Persona(Model):
    id: str
    archetype: str
    title: str = ""
    level: str = ""
    company_shape: str = ""
    speaking_style: str = ""
    known_facts: dict[str, str] = Field(default_factory=dict)


class ScenarioKind(StrEnum):
    ORDINARY = "ordinary"
    EDGE = "edge"
    PRESSURE = "pressure"


class Expectations(Model):
    checks: list[CheckSpec] = Field(default_factory=list)
    judge_criteria: list[Criterion] = Field(default_factory=list)


class Scenario(Model):
    id: str
    goal: str
    kind: ScenarioKind = ScenarioKind.ORDINARY
    opening_message: str
    max_turns: int = Field(default=8, ge=1, le=50)
    expectations: Expectations = Field(default_factory=Expectations)


class TaskFixture(Model):
    kind: Literal["task"] = "task"
    id: str
    input_doc: str
    gold: dict[str, GoldField] = Field(default_factory=dict)
    checks: list[CheckSpec] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @property
    def fixture_hash(self) -> str:
        return content_hash(self)


class ConversationFixture(Model):
    kind: Literal["conversation"] = "conversation"
    id: str
    persona: Persona
    scenario: Scenario
    initial_state: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @property
    def fixture_hash(self) -> str:
        return content_hash(self)


Fixture = Annotated[TaskFixture | ConversationFixture, Field(discriminator="kind")]
