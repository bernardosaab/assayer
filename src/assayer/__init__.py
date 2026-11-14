"""Assayer: durable, budget-guarded evals for LLM tasks and conversational agents."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("assayer")
except PackageNotFoundError:  # pragma: no cover - only when running from a raw checkout
    __version__ = "0.0.0"

__all__ = ["__version__"]
