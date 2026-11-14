"""Run events (Observer pattern): what is happening, as it happens.

The application emits small, typed events; it never knows who listens. The CLI renders them live
(progress, per-attempt results, conversation turns), tests record them, and a future UI could
stream them. Observers must be cheap and must never raise into the run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageStarted:
    stage: str  # "execute" | "check" | "judge" | "grade"
    total: int


@dataclass(frozen=True)
class AttemptStarted:
    case_id: str
    repeat: int


@dataclass(frozen=True)
class AttemptFinished:
    case_id: str
    repeat: int
    ok: bool
    error: str | None = None
    duration_ms: float | None = None


@dataclass(frozen=True)
class TurnEvent:
    """One step of a conversation: a user line, an agent reply, or a tool call and its result."""

    case_id: str
    turn: int
    actor: str  # "user" | "agent" | "tool"
    text: str
    error: str | None = None


@dataclass(frozen=True)
class Judged:
    case_id: str
    repeat: int
    verdict: str | None
    reasks: int
    dropped: int
    cached: bool


@dataclass(frozen=True)
class Graded:
    case_id: str
    repeat: int
    outcome: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Halted:
    reason: str


Event = StageStarted | AttemptStarted | AttemptFinished | TurnEvent | Judged | Graded | Halted


class RunObserver(Protocol):
    def notify(self, event: Event) -> None: ...


class NullObserver:
    def notify(self, event: Event) -> None:  # noqa: ARG002 - deliberately ignores everything
        return None


@dataclass
class RecordingObserver:
    events: list[Event] = field(default_factory=list)

    def notify(self, event: Event) -> None:
        self.events.append(event)

    def of[E](self, kind: type[E]) -> list[E]:
        return [e for e in self.events if isinstance(e, kind)]


def safe_notify(observer: RunObserver, event: Event) -> None:
    """An observer bug must never fail a run."""
    try:
        observer.notify(event)
    except Exception:
        log.exception("run observer failed on %s", type(event).__name__)
