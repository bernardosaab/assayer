"""Name -> factory registries (Registry pattern).

YAML refers to checks and runners by name (``use: field_match``). Built-ins register themselves
on import, and a project can register its own the same way without touching harness code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from assayer.domain.errors import ConfigError
from assayer.domain.fixtures import CheckSpec
from assayer.ports.grading import Check
from assayer.ports.sut import ConversationalAgent, TaskRunner


class Registry[F]:
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._factories: dict[str, F] = {}

    def register(self, name: str) -> Callable[[F], F]:
        def decorator(factory: F) -> F:
            if name in self._factories:
                raise ValueError(f"{self._kind} {name!r} registered twice")
            self._factories[name] = factory
            return factory

        return decorator

    def get(self, name: str) -> F:
        try:
            return self._factories[name]
        except KeyError:
            known = ", ".join(sorted(self._factories)) or "none"
            raise ConfigError(f"unknown {self._kind} {name!r} (known: {known})") from None

    def names(self) -> list[str]:
        return sorted(self._factories)


CheckFactory = Callable[[CheckSpec], Check]
TaskRunnerFactory = Callable[..., TaskRunner]
AgentFactory = Callable[..., ConversationalAgent]

CHECKS: Registry[CheckFactory] = Registry("check")
TASK_RUNNERS: Registry[TaskRunnerFactory] = Registry("task runner")
AGENTS: Registry[AgentFactory] = Registry("agent")


def build_check(spec: CheckSpec) -> Check:
    return CHECKS.get(spec.use)(spec)


def build_task_runner(name: str, **kwargs: Any) -> TaskRunner:
    return TASK_RUNNERS.get(name)(**kwargs)


def build_agent(name: str, **kwargs: Any) -> ConversationalAgent:
    return AGENTS.get(name)(**kwargs)
