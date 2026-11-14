"""The HR assistant eval end to end, offline: scripted models, real environment, real store.

Proves the whole pipeline (conversation loop, tool environment, trajectory checks, judge
protocol, grading, resume) on the scripted fake profile, including its deliberate failures.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from assayer.adapters.fixtures.plugin_loader import load_object
from assayer.adapters.fixtures.yaml_repository import YamlEvalRepository, read_yaml
from assayer.adapters.providers.fake import ScriptedProvider, ScriptRule
from assayer.adapters.store.sqlite_store import SqliteStore
from assayer.adapters.system.clock import FixedClock
from assayer.adapters.system.git import FakeVcs
from assayer.application.chat_model import RoleSettings
from assayer.application.harness import Harness, RoleBinding
from assayer.application.judge import LlmJudge
from assayer.application.provider_middleware import MeteredProvider
from assayer.application.run_eval import RunEval, RunOptions
from assayer.builtin import load_builtins
from assayer.builtin.llm_simulator import LlmUserSimulator
from assayer.builtin.tool_agent import ToolAgent
from assayer.domain.evalspec import EvalSpec
from assayer.domain.outcome import Outcome
from assayer.domain.outputs import Transcript, parse_output
from assayer.domain.pricing import ModelPrice, PriceEntry, PriceTable
from assayer.domain.run import RunStatus

EVAL_DIR = Path(__file__).resolve().parents[2] / "evals" / "hr_assistant"

EXPECTED_OUTCOMES = {
    "book-pto--new-grad": Outcome.PASS,
    "pto-balance--non-native": Outcome.PASS,  # judge's fabricated quote is dropped
    "over-balance--executive": Outcome.PASS,
    "holiday-span--skeptical-manager": Outcome.PASS,
    "colleague-salary--new-grad": Outcome.PASS,
    "raise-promise--skeptical-manager": Outcome.PASS,  # judge re-asked after contradicting itself
    "submit-timeout--non-native": Outcome.FAIL,  # claims success after the write timed out
    "injection-remote-work--executive": Outcome.PASS,
}


def build_harness(spec: EvalSpec, provider: ScriptedProvider, store: SqliteStore) -> Harness:
    assert spec.agent is not None
    assert spec.simulator is not None
    prices = PriceTable(
        entries=[
            PriceEntry(
                pattern="fake/*",
                price=ModelPrice(input_per_mtok=Decimal("0.15"), output_per_mtok=Decimal("0.60")),
            )
        ]
    )
    metered = MeteredProvider(provider, store, prices)

    def role(model: str) -> RoleBinding:
        family = model.replace("/", "-")
        return RoleBinding(provider=metered, settings=RoleSettings(model=model, family=family))

    return Harness(
        spec=spec,
        agent=role("fake/agent"),
        simulator=role("fake/simulator"),
        judge_binding=role("fake/judge"),
        conversational_agent=ToolAgent(max_tool_steps=spec.agent.max_tool_steps),
        user_simulator=LlmUserSimulator(system_prompt=spec.simulator.system_prompt),
        environment_factory=load_object(spec.agent.environment, spec.root)(),
        judge=LlmJudge(spec.judge),
    )


async def test_hr_assistant_eval_runs_offline_end_to_end(tmp_path: Path) -> None:
    load_builtins()
    spec = YamlEvalRepository().load(EVAL_DIR)
    assert spec.fake_script is not None
    assert {c.id for c in spec.cases} == set(EXPECTED_OUTCOMES)
    provider = ScriptedProvider([ScriptRule.model_validate(r) for r in read_yaml(spec.fake_script)])
    store = await SqliteStore.open(tmp_path / "r.sqlite", clock=FixedClock())
    try:
        harness = build_harness(spec, provider, store)
        run = RunEval(store, harness, clock=FixedClock(), vcs=FakeVcs())

        result = await run(RunOptions(run_id="hr-1"))

        assert result.status is RunStatus.COMPLETE
        assert result.execution.harness_errors == 0
        assert result.execution.succeeded == len(EXPECTED_OUTCOMES)
        assert result.scoring is not None
        assert result.scoring.judge_errors == 0

        attempts = {a.attempt_id: a for a in await store.list_attempts("hr-1")}
        grades = await store.list_grades("hr-1", result.meta.grader_version)
        outcomes = {attempt.case_id: g.outcome for attempt, g in grades}
        assert outcomes == EXPECTED_OUTCOMES

        reasons = {attempt.case_id: g.reasons for attempt, g in grades}
        chaos = reasons["submit-timeout--non-native"]
        assert any(r.startswith("check no_false_success:") for r in chaos)
        assert any(r.startswith("judge honest_status:") for r in chaos)
        assert not any("no_request_created" in r for r in chaos)  # the end state is clean

        judgements = {
            attempts[j.attempt_id].case_id: j.verdict for j in await store.list_judgements("hr-1")
        }
        assert judgements["raise-promise--skeptical-manager"].reasks >= 1
        balance = judgements["pto-balance--non-native"]
        assert balance.verdict_raw == "fail"
        assert balance.verdict_final == "pass"
        assert [d.finding.quote for d in balance.dropped] == ["You have 12 days of PTO remaining"]
        assert len(judgements["submit-timeout--non-native"].dropped) == 0

        # Score the write: the end state records what happened, whatever the agent said.
        by_case = {a.case_id: a for a in attempts.values()}
        booked = parse_output(by_case["book-pto--new-grad"].output or {})
        assert isinstance(booked, Transcript)
        assert booked.final_state["final"]["pto_requests"][0]["status"] == "pending_approval"
        chaos_out = parse_output(by_case["submit-timeout--non-native"].output or {})
        assert isinstance(chaos_out, Transcript)
        assert chaos_out.final_state["final"]["pto_requests"] == []
        assert chaos_out.tool_events[0].error is not None

        # Resuming a finished run is free: zero new model calls.
        calls_before = len(provider.calls)
        again = await run(RunOptions(run_id="hr-1"))
        assert again.status is RunStatus.COMPLETE
        assert again.resumed
        assert len(provider.calls) == calls_before
    finally:
        await store.close()


@pytest.mark.parametrize("case_id", sorted(EXPECTED_OUTCOMES))
def test_every_case_is_fully_scripted(case_id: str) -> None:
    """Each case has agent, simulator and judge rules (an unscripted call would fail loudly)."""
    rules = [ScriptRule.model_validate(r) for r in read_yaml(EVAL_DIR / "fake" / "script.yaml")]
    purposes = {
        r.match.purpose.value
        for r in rules
        if r.match.purpose is not None and r.match.case_id == case_id
    }
    assert {"agent", "simulator"} <= purposes
