import pytest

from assayer.domain.attempt import AttemptEvent, AttemptStatus, recover, transition
from assayer.domain.errors import ErrorKind, IllegalTransition

S, E = AttemptStatus, AttemptEvent  # short aliases for the tables below


@pytest.mark.parametrize(
    ("status", "event", "expected"),
    [
        (S.PENDING, E.CLAIM, S.IN_PROGRESS),
        (S.IN_PROGRESS, E.SUCCEED, S.SUCCEEDED),
        (S.IN_PROGRESS, E.ERROR, S.HARNESS_ERROR),
        (S.IN_PROGRESS, E.RESET, S.PENDING),
        (S.HARNESS_ERROR, E.RETRY, S.PENDING),
    ],
)
def test_legal_transitions(
    status: AttemptStatus, event: AttemptEvent, expected: AttemptStatus
) -> None:
    assert transition(status, event) is expected


@pytest.mark.parametrize(
    ("status", "event"),
    [
        (S.SUCCEEDED, E.CLAIM),
        (S.SUCCEEDED, E.RETRY),
        (S.PENDING, E.SUCCEED),
        (S.IN_PROGRESS, E.CLAIM),
    ],
)
def test_illegal_transitions_raise(status: AttemptStatus, event: AttemptEvent) -> None:
    with pytest.raises(IllegalTransition):
        transition(status, event)


def test_recover_uncharged_in_progress_is_reset() -> None:
    assert recover(S.IN_PROGRESS, charged=False) == (S.PENDING, None)


def test_recover_charged_in_progress_becomes_visible_error() -> None:
    assert recover(S.IN_PROGRESS, charged=True) == (
        S.HARNESS_ERROR,
        ErrorKind.INTERRUPTED_AFTER_CHARGE,
    )


@pytest.mark.parametrize("status", [S.PENDING, S.SUCCEEDED, S.HARNESS_ERROR])
def test_recover_leaves_settled_states_alone(status: AttemptStatus) -> None:
    assert recover(status, charged=True) == (status, None)
