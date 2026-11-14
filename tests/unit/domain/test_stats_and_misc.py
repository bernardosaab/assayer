import pytest

from assayer.domain.base import canonical_json, content_hash
from assayer.domain.fixtures import CheckSpec, ConversationFixture, TaskFixture
from assayer.domain.messages import Message, ToolCall
from assayer.domain.outcome import Outcome
from assayer.domain.outputs import TerminationReason, Transcript
from assayer.domain.stats import flips, summarize, wilson_interval


def test_wilson_known_value() -> None:
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4)
    assert hi == pytest.approx(0.9433, abs=1e-4)


def test_wilson_edges() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0
    assert lo > 0.65
    with pytest.raises(ValueError, match="successes"):
        wilson_interval(3, 2)


def test_summary_excludes_harness_errors_from_rate() -> None:
    s = summarize([Outcome.PASS, Outcome.PASS, Outcome.FAIL, Outcome.HARNESS_ERROR])
    assert (s.passed, s.failed, s.harness_errors, s.graded) == (2, 1, 1, 3)
    assert s.pass_rate == pytest.approx(2 / 3)
    assert summarize([Outcome.HARNESS_ERROR]).pass_rate is None


def test_flips_ignore_harness_errors() -> None:
    base = {"a": Outcome.PASS, "b": Outcome.FAIL, "c": Outcome.PASS, "d": Outcome.PASS}
    cand = {"a": Outcome.FAIL, "b": Outcome.PASS, "c": Outcome.HARNESS_ERROR, "d": Outcome.PASS}
    assert [(f.key, f.candidate) for f in flips(base, cand)] == [
        ("a", Outcome.FAIL),
        ("b", Outcome.PASS),
    ]


def test_canonical_json_and_hash_are_order_independent() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert content_hash({"b": 1, "a": 2}) == content_hash({"a": 2, "b": 1})
    assert content_hash("x") != content_hash("y")


def test_fixture_hash_changes_with_content() -> None:
    a = TaskFixture(id="c1", input_doc="doc", gold={"name": "Ann"})
    b = TaskFixture(id="c1", input_doc="doc2", gold={"name": "Ann"})
    assert a.fixture_hash != b.fixture_hash


def test_conversation_fixture_parses_nested() -> None:
    fx = ConversationFixture.model_validate(
        {
            "id": "c",
            "persona": {"id": "p", "archetype": "new grad"},
            "scenario": {"id": "s", "goal": "book PTO", "opening_message": "hi"},
        }
    )
    assert fx.scenario.max_turns == 8
    assert CheckSpec(use="x").check_id == "x"
    assert CheckSpec(use="x", id="y").check_id == "y"


def test_transcript_render_skips_system_and_shows_tools() -> None:
    t = Transcript(
        messages=(
            Message.system("secret"),
            Message.user("hello"),
            Message.assistant(None, (ToolCall(id="1", name="get_pto_balance", arguments={}),)),
            Message.tool("1", '{"days": 3}'),
            Message.assistant("You have 3 days."),
        ),
        termination=TerminationReason.GOAL_MET,
    )
    text = t.render()
    assert "secret" not in text
    assert "USER: hello" in text
    assert "[tool call] get_pto_balance" in text
    assert "AGENT: You have 3 days." in text
