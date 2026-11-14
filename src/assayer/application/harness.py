"""The assembled harness for one eval: the SUT plus a model binding per role.

Built by the composition root (``cli``), consumed by the use cases. Keeping construction out of
the application layer is what lets tests assemble a harness from fakes in a few lines.
"""

from __future__ import annotations

from dataclasses import dataclass

from assayer.application.chat_model import BoundChatModel, RoleSettings
from assayer.domain.evalspec import EvalSpec
from assayer.domain.messages import CallContext, Purpose
from assayer.ports.grading import Judge
from assayer.ports.provider import ModelProvider
from assayer.ports.sut import ConversationalAgent, EnvironmentFactory, TaskRunner, UserSimulator


@dataclass(frozen=True)
class RoleBinding:
    provider: ModelProvider  # already wrapped in middleware (metering, retries...)
    settings: RoleSettings

    def bind(self, context: CallContext) -> BoundChatModel:
        return BoundChatModel(self.provider, self.settings, context)


@dataclass(frozen=True)
class Harness:
    spec: EvalSpec
    agent: RoleBinding
    simulator: RoleBinding | None = None
    judge_binding: RoleBinding | None = None
    task_runner: TaskRunner | None = None
    conversational_agent: ConversationalAgent | None = None
    environment_factory: EnvironmentFactory | None = None
    user_simulator: UserSimulator | None = None
    judge: Judge | None = None

    @property
    def judge_version(self) -> str | None:
        model = self.judge_binding.settings.model if self.judge_binding else None
        return self.spec.judge_version(model)

    def context(self, purpose: Purpose, run_id: str, case_id: str, attempt_id: int) -> CallContext:
        return CallContext(purpose=purpose, run_id=run_id, case_id=case_id, attempt_id=attempt_id)
