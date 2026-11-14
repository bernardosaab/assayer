"""Load eval-local Python (e.g. an agent's tool environment) from ``file.py:Name``.

Evals live outside the installed package, so their code is loaded by file path rather than
requiring ``evals/`` to be an importable package. Only files inside the eval directory are
allowed; this runs the eval author's own code, the same trust level as a test file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from assayer.domain.errors import ConfigError


def load_object(reference: str, root: Path) -> Any:
    file_part, _, name = reference.partition(":")
    if not name:
        raise ConfigError(f"expected 'file.py:Name', got {reference!r}")
    path = (root / file_part).resolve()
    if root.resolve() not in path.parents:
        raise ConfigError(f"{reference!r} must be inside the eval directory {root}")
    if not path.is_file():
        raise ConfigError(f"file not found: {path}")
    module_name = f"assayer_eval_{root.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    try:
        return getattr(module, name)
    except AttributeError:
        raise ConfigError(f"{path} has no attribute {name!r}") from None
