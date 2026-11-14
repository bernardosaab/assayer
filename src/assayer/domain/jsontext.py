"""Parse a JSON object out of model text, tolerating a Markdown code fence."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_json_object(raw: str) -> dict[str, Any]:
    """Return the JSON object in ``raw``.

    Raises:
        ValueError: if the text is not a JSON object (``json.JSONDecodeError`` included).
    """
    match = _FENCE.match(raw)
    value = json.loads(match.group(1) if match else raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object, got {type(value).__name__}")
    return value
