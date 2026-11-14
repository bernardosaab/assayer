"""``assayer.toml``: where results go, budgets, and model profiles per role.

A *profile* (``fake``, ``ollama``, ``groq``...) names one model per role: agent, simulator and
judge. The fake profile is built in, so a fresh clone runs offline with no config at all.
"""

from __future__ import annotations

import tomllib
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from assayer.domain.errors import ConfigError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RoleProfile(_Strict):
    provider: Literal["openai_compat", "fake"]
    model: str
    family: str = "unknown"
    base_url: str | None = None
    api_key_env: str | None = None
    max_tokens: int = Field(default=1024, gt=0)
    temperature: float = 0.0
    seed: int | None = None
    concurrency: int = Field(default=4, ge=1)
    timeout_s: float = Field(default=120.0, gt=0)


class Profile(_Strict):
    agent: RoleProfile
    simulator: RoleProfile | None = None
    judge: RoleProfile | None = None


class Defaults(_Strict):
    concurrency: int = Field(default=4, ge=1)
    repeats: int = Field(default=1, ge=1)
    budget_usd: Decimal = Field(default=Decimal("1.00"), ge=0)
    max_retries: int = Field(default=3, ge=0)
    max_tries: int = Field(default=3, ge=1)
    circuit_breaker: int = Field(default=5, ge=1)
    strict_judge_family: bool = False


def _fake_role(model: str, family: str) -> RoleProfile:
    return RoleProfile(provider="fake", model=model, family=family, concurrency=8)


BUILTIN_PROFILES: dict[str, Profile] = {
    "fake": Profile(
        agent=_fake_role("fake/agent", "fake-agent"),
        simulator=_fake_role("fake/simulator", "fake-agent"),
        judge=_fake_role("fake/judge", "fake-judge"),
    )
}


class Settings(_Strict):
    store_path: Path = Path(".assayer/results.sqlite")
    prices_path: Path = Path("prices.yaml")
    reports_dir: Path = Path("reports")
    defaults: Defaults = Field(default_factory=Defaults)
    profiles: dict[str, Profile] = Field(default_factory=dict)

    def profile(self, name: str) -> Profile:
        merged = {**BUILTIN_PROFILES, **self.profiles}
        if name not in merged:
            raise ConfigError(f"unknown profile {name!r} (known: {', '.join(sorted(merged))})")
        return merged[name]


def load_settings(path: Path | None) -> Settings:
    """Load settings; a missing file yields the defaults (offline fake profile only)."""
    if path is None or not path.exists():
        return Settings()
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    base = path.parent
    try:
        settings = Settings.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid settings in {path}:\n{exc}") from exc
    # Relative paths are relative to the config file, not the working directory.
    return settings.model_copy(
        update={
            "store_path": base / settings.store_path,
            "prices_path": base / settings.prices_path,
            "reports_dir": base / settings.reports_dir,
        }
    )
