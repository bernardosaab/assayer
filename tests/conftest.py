"""Shared test configuration.

* A network guard: any test not marked ``live`` that opens a non-loopback socket fails.
  CI must never talk to a real model (ADR-0014).
* Hypothesis profiles: ``dev`` (fast) and ``ci`` (thorough), picked by ``HYPOTHESIS_PROFILE``.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from typing import Any

import pytest
from hypothesis import HealthCheck, settings

settings.register_profile("dev", max_examples=50, deadline=None)
settings.register_profile(
    "ci", max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
_real_connect = socket.socket.connect


@pytest.fixture(autouse=True)
def _network_guard(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    if request.node.get_closest_marker("live"):
        yield
        return

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in _LOOPBACK:
            raise RuntimeError(f"Network access blocked in tests (attempted {address!r})")
        return _real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    yield


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite golden report files instead of comparing against them.",
    )


@pytest.fixture
def update_golden(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-golden"))
