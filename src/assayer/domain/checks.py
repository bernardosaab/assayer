"""Result of a deterministic check."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from assayer.domain.base import Model


class CheckResult(Model):
    check_id: str
    version: str
    passed: bool
    detail: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(cls, check_id: str, version: str, **detail: Any) -> CheckResult:
        return cls(check_id=check_id, version=version, passed=True, detail=detail)

    @classmethod
    def failed(cls, check_id: str, version: str, reason: str, **detail: Any) -> CheckResult:
        return cls(
            check_id=check_id, version=version, passed=False, detail={"reason": reason, **detail}
        )
