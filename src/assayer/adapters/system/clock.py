from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


@dataclass
class FixedClock:
    """Deterministic clock for tests and golden reports. Advances ``step`` per call."""

    start: datetime = field(default_factory=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    step: timedelta = timedelta(0)

    def now(self) -> datetime:
        current = self.start
        self.start = self.start + self.step
        return current
