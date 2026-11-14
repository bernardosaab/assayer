"""Test-data builders (Builder pattern): valid defaults, override only what a test cares about."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from assayer.domain.fixtures import ConversationFixture, TaskFixture
from assayer.domain.money import Money
from assayer.domain.run import ModelRoles, RunMeta


def a_run(run_id: str = "run-1", **overrides: Any) -> RunMeta:
    values: dict[str, Any] = {
        "run_id": run_id,
        "eval_name": "demo",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "prompt_version": "p1",
        "grader_version": "g1",
        "models": ModelRoles(agent="fake/agent"),
        "repeats": 1,
        "budget_micros": Money.from_usd("1.00").micros,
        "config_hash": "cfg-1",
    }
    values.update(overrides)
    return RunMeta(**values)


def a_task(case_id: str = "case-1", **overrides: Any) -> TaskFixture:
    values: dict[str, Any] = {
        "id": case_id,
        "input_doc": "Jane Doe. Senior Engineer at Acme Corp since 2020.",
        "gold": {"name": "Jane Doe", "current_employer": "Acme Corp"},
    }
    values.update(overrides)
    return TaskFixture.model_validate(values)


def a_conversation(case_id: str = "conv-1", **overrides: Any) -> ConversationFixture:
    values: dict[str, Any] = {
        "id": case_id,
        "persona": {"id": "new-grad", "archetype": "new grad engineer"},
        "scenario": {"id": "book-pto", "goal": "Book 2 days of PTO", "opening_message": "Hi!"},
    }
    values.update(overrides)
    return ConversationFixture.model_validate(values)
