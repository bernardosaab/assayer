"""Shared plumbing for built-in checks."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, ClassVar

from assayer.domain.base import content_hash
from assayer.domain.checks import CheckResult
from assayer.domain.fixtures import CheckSpec, ConversationFixture, TaskFixture
from assayer.domain.outputs import TaskOutput, Transcript

Case = TaskFixture | ConversationFixture
Output = TaskOutput | Transcript


class SpecCheck:
    """Base for checks configured by a :class:`CheckSpec`.

    The version combines the implementation's ``LOGIC_VERSION`` with a hash of the arguments,
    so editing either one invalidates stored results for exactly this check (ADR-0009).
    """

    LOGIC_VERSION: ClassVar[str] = "1"

    def __init__(self, spec: CheckSpec) -> None:
        self.spec = spec
        self.args: dict[str, Any] = dict(spec.args)

    @property
    def check_id(self) -> str:
        return self.spec.check_id

    @property
    def version(self) -> str:
        return f"{self.LOGIC_VERSION}-{content_hash(self.spec.use, self.args, length=8)}"

    def ok(self, **detail: Any) -> CheckResult:
        return CheckResult.ok(self.check_id, self.version, **detail)

    def fail(self, reason: str, **detail: Any) -> CheckResult:
        return CheckResult.failed(self.check_id, self.version, reason, **detail)


_WS = re.compile(r"\s+")


def norm_text(value: Any) -> str:
    """Lenient comparison form for scalar values: NFKC, casefold, collapsed whitespace."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return _WS.sub(" ", text).strip().rstrip(".")


def norm_value(value: Any) -> Any:
    """Normalize scalars and lists of scalars (lists compared as sets)."""
    if isinstance(value, list):
        return frozenset(norm_text(v) for v in value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float):
        return float(value)
    return norm_text(value)


def lookup(data: Any, path: str) -> tuple[bool, Any]:
    """Resolve a dotted path (``a.b.0.c``) in nested dicts/lists; ``(found, value)``."""
    current = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, current
