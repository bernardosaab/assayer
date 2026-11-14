"""Assemble a task harness from fakes for integration tests (a mini composition root)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from assayer.adapters.fixtures.yaml_repository import YamlEvalRepository, read_yaml
from assayer.adapters.providers.fake import ScriptedProvider, ScriptRule
from assayer.application.chat_model import RoleSettings
from assayer.application.harness import Harness, RoleBinding
from assayer.application.judge import LlmJudge
from assayer.application.provider_middleware import MeteredProvider
from assayer.application.registry import build_task_runner
from assayer.builtin import load_builtins
from assayer.domain.evalspec import EvalSpec
from assayer.domain.pricing import ModelPrice, PriceEntry, PriceTable
from assayer.ports.provider import ModelProvider
from assayer.ports.store import CostLedger

REPO = Path(__file__).resolve().parents[2]

PRICES = PriceTable(
    entries=[
        PriceEntry(
            pattern="fake/*",
            price=ModelPrice(input_per_mtok=Decimal("0.15"), output_per_mtok=Decimal("0.60")),
        )
    ]
)


def load_spec(name: str) -> EvalSpec:
    load_builtins()
    return YamlEvalRepository().load(REPO / "evals" / name)


def scripted(spec: EvalSpec) -> ScriptedProvider:
    assert spec.fake_script is not None
    return ScriptedProvider([ScriptRule.model_validate(r) for r in read_yaml(spec.fake_script)])


def task_harness(spec: EvalSpec, raw: ModelProvider, ledger: CostLedger) -> Harness:
    assert spec.task is not None
    metered = MeteredProvider(raw, ledger, PRICES, default_max_tokens=512)
    judge_binding = (
        RoleBinding(metered, RoleSettings(model="fake/judge", family="fake-judge"))
        if spec.judge
        else None
    )
    return Harness(
        spec=spec,
        agent=RoleBinding(metered, RoleSettings(model="fake/agent", family="fake-agent")),
        judge_binding=judge_binding,
        judge=LlmJudge(spec.judge) if spec.judge else None,
        task_runner=build_task_runner(
            spec.task.runner,
            system_prompt=spec.task.system_prompt,
            output_schema=spec.task.output_schema,
            **spec.task.args,
        ),
    )
