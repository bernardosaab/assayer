"""Shared Pydantic base for immutable domain values."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict


class Model(BaseModel):
    """Immutable, strict-schema value object."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def canonical_json(value: Any) -> str:
    """Deterministic JSON (sorted keys, fixed separators). Used for hashing and storage."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(*parts: Any, length: int = 16) -> str:
    """Short, stable SHA-256 over the canonical JSON of ``parts``."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(canonical_json(part).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:length]
