"""Built-in task runners, agents, simulators and checks.

Importing :func:`load_builtins` registers them in the application registries. The composition
root calls it once; nothing else in the harness depends on this package.
"""

from __future__ import annotations


def load_builtins() -> None:
    # Imported for their registration side effects.
    import assayer.builtin.checks.combinators
    import assayer.builtin.checks.fields
    import assayer.builtin.checks.trajectory
    import assayer.builtin.extraction_task
    import assayer.builtin.tool_agent  # noqa: F401
