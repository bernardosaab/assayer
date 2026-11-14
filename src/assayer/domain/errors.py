"""Error taxonomy.

The central distinction (ADR-0008): a *harness error* means the attempt could not be measured
because infrastructure failed. It is excluded from every rate and the attempt is re-run. Bad
output from the agent itself is **not** a harness error; it is graded ``fail``.
"""

from __future__ import annotations

from enum import StrEnum

from assayer.domain.money import Money


class ErrorKind(StrEnum):
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_RATE_LIMITED = "provider_rate_limited"
    PROVIDER_SERVER_ERROR = "provider_server_error"
    PROVIDER_BAD_REQUEST = "provider_bad_request"
    PROVIDER_INVALID_RESPONSE = "provider_invalid_response"
    STORE_LOCKED = "store_locked"
    INTERRUPTED_AFTER_CHARGE = "interrupted_after_charge"
    JUDGE_INCONSISTENT = "judge_inconsistent"
    JUDGE_UNPARSEABLE = "judge_unparseable"
    BUDGET_EXCEEDED = "budget_exceeded"
    INTERNAL = "internal"


class AssayerError(Exception):
    """Base class for all errors raised by assayer."""


class HarnessError(AssayerError):
    """Infrastructure failure: the attempt could not be measured."""

    kind: ErrorKind = ErrorKind.INTERNAL

    def __init__(self, message: str, *, kind: ErrorKind | None = None) -> None:
        super().__init__(message)
        if kind is not None:
            self.kind = kind


class ProviderError(HarnessError):
    """A model call failed.

    ``request_sent`` tells the cost guard whether the provider may have billed the call. If the
    request never left (connection refused, 429 rejection) the reservation is released;
    otherwise the reserved worst case is kept (ADR-0006).
    """

    retryable: bool = False
    request_sent: bool = True


class ProviderUnavailable(ProviderError):
    kind = ErrorKind.PROVIDER_UNAVAILABLE
    retryable = True
    request_sent = False


class ProviderTimeout(ProviderError):
    kind = ErrorKind.PROVIDER_TIMEOUT
    retryable = True


class ProviderRateLimited(ProviderError):
    kind = ErrorKind.PROVIDER_RATE_LIMITED
    retryable = True
    request_sent = False

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ProviderServerError(ProviderError):
    kind = ErrorKind.PROVIDER_SERVER_ERROR
    retryable = True


class ProviderBadRequest(ProviderError):
    kind = ErrorKind.PROVIDER_BAD_REQUEST
    request_sent = False


class ProviderInvalidResponse(ProviderError):
    kind = ErrorKind.PROVIDER_INVALID_RESPONSE


class StoreLocked(HarnessError):
    kind = ErrorKind.STORE_LOCKED


class BudgetExceeded(HarnessError):
    """Raised *before* a call whose worst case would push committed spend past the budget."""

    kind = ErrorKind.BUDGET_EXCEEDED

    def __init__(self, *, committed: Money, requested: Money, budget: Money) -> None:
        super().__init__(
            f"Budget exceeded: committed {committed} + requested {requested} > budget {budget}"
        )
        self.committed = committed
        self.requested = requested
        self.budget = budget


class UnresolvedGold(AssayerError):
    """A gold field has disagreeing labeller votes and no resolution."""


class ConfigError(AssayerError):
    """Invalid configuration or eval definition."""


class IllegalTransition(AssayerError):
    """An attempt was asked to make a transition the state machine forbids."""
