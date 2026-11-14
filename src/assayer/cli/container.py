"""Composition root: the one place that knows every concrete class.

It reads settings and the eval definition, builds providers for the chosen profile, wraps them
in middleware, and assembles the :class:`Harness`. Everything else receives collaborators
through constructors (dependency injection), which is what keeps the core testable with fakes.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from assayer.adapters.fixtures.plugin_loader import load_object
from assayer.adapters.fixtures.yaml_repository import YamlEvalRepository, read_yaml
from assayer.adapters.pricing.yaml_price_table import load_price_table
from assayer.adapters.providers.fake import ScriptedProvider, ScriptRule
from assayer.adapters.providers.faults import FaultInjectingProvider
from assayer.adapters.providers.openai_compat import OpenAICompatProvider
from assayer.adapters.store.sqlite_store import SqliteStore
from assayer.adapters.system.clock import SystemClock
from assayer.adapters.system.git import GitVcs
from assayer.application.chat_model import RoleSettings
from assayer.application.harness import Harness, RoleBinding
from assayer.application.judge import LlmJudge
from assayer.application.provider_middleware import (
    LimitedProvider,
    MeteredProvider,
    RetryingProvider,
    TimeoutProvider,
)
from assayer.application.registry import build_agent, build_task_runner
from assayer.builtin import load_builtins
from assayer.builtin.llm_simulator import LlmUserSimulator
from assayer.config import Defaults, RoleProfile, Settings
from assayer.domain.errors import ConfigError
from assayer.domain.evalspec import EvalKind, EvalSpec
from assayer.domain.pricing import ModelPrice, PriceTable
from assayer.ports.provider import ModelProvider
from assayer.ports.system import Clock, VcsInfo

log = logging.getLogger(__name__)

# Test hook for the crash/resume end-to-end test: kill the process on the Nth model call.
CRASH_ENV = "ASSAYER_TEST_HARD_EXIT_ON_CALL"


@dataclass
class Container:
    settings: Settings
    store: SqliteStore
    clock: Clock
    vcs: VcsInfo
    harness: Harness | None = None
    _closers: list[Callable[[], Any]] = field(default_factory=list)

    async def aclose(self) -> None:
        for closer in reversed(self._closers):
            result = closer()
            if hasattr(result, "__await__"):
                await result
        await self.store.close()


async def open_store(settings: Settings, clock: Clock) -> SqliteStore:
    return await SqliteStore.open(settings.store_path, clock=clock)


def load_eval(eval_ref: str | Path) -> EvalSpec:
    """Accept a path to an eval directory or a bare name under ``evals/``."""
    path = Path(eval_ref)
    if not (path / "eval.yaml").exists() and (Path("evals") / path / "eval.yaml").exists():
        path = Path("evals") / path
    load_builtins()
    return YamlEvalRepository().load(path)


def _prices(settings: Settings, allow_unpriced: bool) -> PriceTable:
    table = (
        load_price_table(settings.prices_path) if settings.prices_path.exists() else PriceTable()
    )
    if allow_unpriced:
        log.warning(
            "--allow-unpriced: unknown models are treated as free; budgets cannot protect them"
        )
        table = table.with_fallback(
            ModelPrice(input_per_mtok=Decimal(0), output_per_mtok=Decimal(0))
        )
    return table


class _ProviderFactory:
    """Builds one raw provider per distinct endpoint and shares it across roles."""

    def __init__(self, spec: EvalSpec, container: Container) -> None:
        self._spec = spec
        self._container = container
        self._cache: dict[tuple[str, str | None], ModelProvider] = {}

    def raw(self, role: RoleProfile) -> ModelProvider:
        key = (role.provider, role.base_url)
        if key in self._cache:
            return self._cache[key]
        provider: ModelProvider
        if role.provider == "fake":
            if self._spec.fake_script is None:
                raise ConfigError(
                    f"eval {self._spec.name!r} has no fake/script.yaml for --profile fake"
                )
            rules = [ScriptRule.model_validate(r) for r in read_yaml(self._spec.fake_script) or []]
            provider = ScriptedProvider(rules)
            if crash_on := os.environ.get(CRASH_ENV):
                provider = FaultInjectingProvider(provider, hard_exit_on_call=int(crash_on))
        else:
            if role.base_url is None:
                raise ConfigError(f"model {role.model!r}: openai_compat needs base_url")
            api_key = os.environ.get(role.api_key_env) if role.api_key_env else None
            if role.api_key_env and not api_key:
                raise ConfigError(f"environment variable {role.api_key_env} is not set")
            http = OpenAICompatProvider(
                name=role.model.split("/", 1)[0], base_url=role.base_url, api_key=api_key
            )
            self._container._closers.append(http.aclose)
            provider = http
        self._cache[key] = provider
        return provider


def _bind(
    role: RoleProfile,
    raw: ModelProvider,
    container: Container,
    prices: PriceTable,
    defaults: Defaults,
) -> RoleBinding:
    wrapped = RetryingProvider(
        LimitedProvider(
            MeteredProvider(
                TimeoutProvider(raw, timeout_s=role.timeout_s),
                container.store,
                prices,
                default_max_tokens=role.max_tokens,
            ),
            max_concurrency=role.concurrency,
        ),
        max_retries=defaults.max_retries,
    )
    settings = RoleSettings(
        model=role.model,
        family=role.family,
        max_tokens=role.max_tokens,
        temperature=role.temperature,
        seed=role.seed,
    )
    return RoleBinding(provider=wrapped, settings=settings)


def check_judge_family(
    agent: RoleProfile, judge: RoleProfile | None, *, strict: bool
) -> str | None:
    """A judge from the agent's own model family tends to favour it (ADR-0012)."""
    if judge is None or judge.family != agent.family or agent.family == "unknown":
        return None
    message = (
        f"judge model {judge.model!r} is from the same family as the agent ({agent.family}); "
        "self-preference bias is likely. Prefer a judge from another family."
    )
    if strict:
        raise ConfigError(message)
    return message


async def build_container(
    settings: Settings,
    *,
    eval_ref: str | Path | None = None,
    profile_name: str = "fake",
    allow_unpriced: bool = False,
    warn: Callable[[str], None] | None = None,
) -> Container:
    clock = SystemClock()
    container = Container(
        settings=settings,
        store=await open_store(settings, clock),
        clock=clock,
        vcs=GitVcs(Path.cwd()),
    )
    if eval_ref is None:
        return container
    try:
        spec = load_eval(eval_ref)
        container.harness = _assemble(
            spec=spec,
            settings=settings,
            profile_name=profile_name,
            container=container,
            allow_unpriced=allow_unpriced,
            warn=warn,
        )
    except BaseException:
        await container.aclose()
        raise
    return container


def _assemble(
    *,
    spec: EvalSpec,
    settings: Settings,
    profile_name: str,
    container: Container,
    allow_unpriced: bool,
    warn: Callable[[str], None] | None,
) -> Harness:
    profile = settings.profile(profile_name)
    prices = _prices(
        settings, allow_unpriced or (profile_name == "fake" and not settings.prices_path.exists())
    )
    factory = _ProviderFactory(spec, container)

    def bind(role: RoleProfile) -> RoleBinding:
        return _bind(role, factory.raw(role), container, prices, settings.defaults)

    if spec.judge is not None and profile.judge is not None:
        note = check_judge_family(
            profile.agent, profile.judge, strict=settings.defaults.strict_judge_family
        )
        if note and warn:
            warn(note)

    agent = bind(profile.agent)
    judge_binding = bind(profile.judge) if spec.judge and profile.judge else None
    judge = LlmJudge(spec.judge) if judge_binding else None

    if spec.kind is EvalKind.TASK:
        assert spec.task is not None
        runner = build_task_runner(
            spec.task.runner,
            system_prompt=spec.task.system_prompt,
            output_schema=spec.task.output_schema,
            **spec.task.args,
        )
        return Harness(
            spec=spec, agent=agent, judge_binding=judge_binding, judge=judge, task_runner=runner
        )

    if spec.agent is None or spec.simulator is None:
        raise ConfigError(f"{spec.name}: conversation evals need 'agent' and 'simulator'")
    sim_role = profile.simulator or profile.agent
    env_factory_cls = load_object(spec.agent.environment, spec.root)
    return Harness(
        spec=spec,
        agent=agent,
        simulator=bind(sim_role),
        judge_binding=judge_binding,
        judge=judge,
        conversational_agent=build_agent(
            spec.agent.runner, max_tool_steps=spec.agent.max_tool_steps, **spec.agent.args
        ),
        environment_factory=env_factory_cls(),
        user_simulator=LlmUserSimulator(system_prompt=spec.simulator.system_prompt),
    )
