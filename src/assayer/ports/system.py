"""Ports onto the outside world that are not models or storage."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class VcsInfo(Protocol):
    def sha(self) -> str | None: ...
    def dirty(self) -> bool: ...
