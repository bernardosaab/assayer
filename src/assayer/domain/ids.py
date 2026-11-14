"""Typed identifiers. ``NewType`` keeps them zero-cost at runtime but distinct for mypy."""

from typing import NewType

RunId = NewType("RunId", str)
CaseId = NewType("CaseId", str)
AttemptId = NewType("AttemptId", int)
