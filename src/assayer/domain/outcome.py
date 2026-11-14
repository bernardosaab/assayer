"""The three outcomes of a graded attempt (ADR-0008)."""

from enum import StrEnum


class Outcome(StrEnum):
    PASS = "pass"  # noqa: S105 - not a password
    FAIL = "fail"
    HARNESS_ERROR = "harness_error"
