"""Run metadata: what was run, with which code, prompts, models and graders (observability)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from assayer.domain.base import Model


class RunStatus(StrEnum):
    OPEN = "open"
    HALTED = "halted"
    COMPLETE = "complete"


class ModelRoles(Model):
    agent: str
    simulator: str | None = None
    judge: str | None = None


class RunMeta(Model):
    run_id: str
    eval_name: str
    created_at: datetime
    git_sha: str | None = None
    git_dirty: bool = False
    prompt_version: str
    judge_version: str | None = None
    grader_version: str
    models: ModelRoles
    repeats: int = Field(ge=1)
    round: int = Field(default=1, ge=1)
    budget_micros: int = Field(ge=0)
    config_hash: str
    replay_of: str | None = None
