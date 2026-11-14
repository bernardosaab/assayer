"""Git metadata for run observability. Never fails a run: missing git just yields ``None``."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitVcs:
    cwd: Path

    def _git(self, *args: str) -> str | None:
        exe = shutil.which("git")
        if exe is None:
            return None
        try:
            done = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [exe, *args], cwd=self.cwd, capture_output=True, text=True, timeout=5, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    def sha(self) -> str | None:
        return self._git("rev-parse", "HEAD")

    def dirty(self) -> bool:
        status = self._git("status", "--porcelain")
        return bool(status)


@dataclass(frozen=True)
class FakeVcs:
    fixed_sha: str | None = "0" * 40
    is_dirty: bool = False

    def sha(self) -> str | None:
        return self.fixed_sha

    def dirty(self) -> bool:
        return self.is_dirty
