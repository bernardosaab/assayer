"""Enforce the hexagonal layering (ADR-0002) with a plain AST scan.

Allowed dependency directions (an arrow means "may import"):

    cli -> everything
    builtin -> application, ports, domain
    adapters -> ports, domain
    application -> ports, domain
    ports -> domain
    domain -> (stdlib + pydantic only)
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "assayer"

FORBIDDEN: dict[str, set[str]] = {
    "domain": {"ports", "application", "adapters", "builtin", "cli", "config"},
    "ports": {"application", "adapters", "builtin", "cli", "config"},
    "application": {"adapters", "builtin", "cli", "config"},
    "adapters": {"application", "builtin", "cli"},
    "builtin": {"adapters", "cli"},
}

# The domain must stay free of I/O and frameworks.
DOMAIN_FORBIDDEN_THIRD_PARTY = {"httpx", "sqlite3", "yaml", "jinja2", "typer", "asyncio", "rich"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def _layer_files(layer: str) -> list[Path]:
    return sorted((SRC / layer).rglob("*.py"))


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_layer_does_not_import_outer_layers(layer: str) -> None:
    violations: list[str] = []
    for path in _layer_files(layer):
        for module in _imports(path):
            parts = module.split(".")
            if parts[0] == "assayer" and len(parts) > 1 and parts[1] in FORBIDDEN[layer]:
                violations.append(f"{path.relative_to(SRC)} imports {module}")
    assert not violations, "Layering violations:\n" + "\n".join(violations)


def test_domain_is_pure() -> None:
    violations = [
        f"{path.relative_to(SRC)} imports {module}"
        for path in _layer_files("domain")
        for module in _imports(path)
        if module.split(".")[0] in DOMAIN_FORBIDDEN_THIRD_PARTY
    ]
    assert not violations, "Domain must be I/O-free:\n" + "\n".join(violations)
