"""A tiny forward-only migrator: numbered ``NNNN_name.sql`` files applied in order.

Each migration runs in its own ``BEGIN IMMEDIATE`` transaction, and the current version is
re-read *inside* that transaction, so two processes opening a fresh database at once cannot
both apply the same migration.
"""

from __future__ import annotations

import re
import sqlite3
from importlib import resources

_NAME = re.compile(r"^(\d{4})_[\w-]+\.sql$")


def available_migrations() -> list[tuple[int, str]]:
    root = resources.files("assayer.adapters.store.migrations")
    found: list[tuple[int, str]] = []
    for entry in root.iterdir():
        match = _NAME.match(entry.name)
        if match:
            found.append((int(match.group(1)), entry.read_text(encoding="utf-8")))
    return sorted(found)


def split_statements(script: str) -> list[str]:
    """Split a SQL script into complete statements using SQLite's own tokenizer check."""
    statements: list[str] = []
    buffer = ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            if buffer.strip():
                statements.append(buffer.strip())
            buffer = ""
    if buffer.strip() and not all(
        ln.strip().startswith("--") or not ln.strip() for ln in buffer.splitlines()
    ):
        raise ValueError(f"incomplete SQL statement at end of migration: {buffer[:80]!r}")
    return statements


def current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()
    return int(row[0])


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations; return the resulting schema version."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version "
        "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    for version, script in available_migrations():
        conn.execute("BEGIN IMMEDIATE")
        try:
            if version <= current_version(conn):
                conn.execute("ROLLBACK")
                continue
            for statement in split_statements(script):
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, datetime('now'))",
                (version,),
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    return current_version(conn)
