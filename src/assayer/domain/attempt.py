"""The attempt state machine (ADR-0004, ADR-0005).

An *attempt* is one execution of one case: ``(run_id, case_id, repeat)``. It is the unit of
durability. Its row is created and claimed (``in_progress``) *before* the first model call, so
a crash always leaves a trace.

::

    pending ──claim──► in_progress ──succeed──► succeeded
       ▲                  │   │
       └──reset (no spend)┘   └──error──► harness_error ──retry──► pending

On resume, an ``in_progress`` attempt with no spend is ``reset`` and redone. One with spend is
moved to ``harness_error`` (``interrupted_after_charge``): the money is already gone, so the
attempt is surfaced instead of silently re-spent.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from assayer.domain.base import Model
from assayer.domain.errors import ErrorKind, IllegalTransition


class AttemptStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    HARNESS_ERROR = "harness_error"


class AttemptEvent(StrEnum):
    CLAIM = "claim"
    SUCCEED = "succeed"
    ERROR = "error"
    RESET = "reset"
    RETRY = "retry"


_S, _E = AttemptStatus, AttemptEvent

TRANSITIONS: dict[tuple[AttemptStatus, AttemptEvent], AttemptStatus] = {
    (_S.PENDING, _E.CLAIM): _S.IN_PROGRESS,
    (_S.IN_PROGRESS, _E.SUCCEED): _S.SUCCEEDED,
    (_S.IN_PROGRESS, _E.ERROR): _S.HARNESS_ERROR,
    (_S.IN_PROGRESS, _E.RESET): _S.PENDING,
    (_S.HARNESS_ERROR, _E.RETRY): _S.PENDING,
}


def transition(status: AttemptStatus, event: AttemptEvent) -> AttemptStatus:
    try:
        return TRANSITIONS[(status, event)]
    except KeyError:
        raise IllegalTransition(f"cannot {event.value} an attempt that is {status.value}") from None


def recover(status: AttemptStatus, *, charged: bool) -> tuple[AttemptStatus, ErrorKind | None]:
    """Decide what an attempt found on resume becomes.

    Only ``in_progress`` attempts need recovery (they were interrupted). Everything else is
    returned unchanged.
    """
    if status is not AttemptStatus.IN_PROGRESS:
        return status, None
    if charged:
        return transition(status, AttemptEvent.ERROR), ErrorKind.INTERRUPTED_AFTER_CHARGE
    return transition(status, AttemptEvent.RESET), None


class Attempt(Model):
    attempt_id: int
    run_id: str
    case_id: str
    repeat: int = Field(ge=0)
    status: AttemptStatus
    output: dict[str, Any] | None = None
    error_kind: ErrorKind | None = None
    error_msg: str | None = None
    cost_micros: int = 0
    try_count: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def duration_ms(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds() * 1000
