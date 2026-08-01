"""Agent memory.

Persistent memory that lets the agent improve across runs. Today this stores
learned label aliases (seed: ``atlas.mapping``) so that a label the user or
planner confirms once ("DOB" -> "date of birth") is remembered for every later
record and session. The store is a plain SQLite file - no server, no state
machines, trivially inspectable.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from atlas.core.logging import logger

_SCHEMA = """
CREATE TABLE IF NOT EXISTS aliases (
    variant      TEXT PRIMARY KEY,
    canonical    TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
    hits         INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class MemoryStore:
    """SQLite-backed persistent memory.

    Thread-safe for the observer / executor threads that share the process.
    ``in_memory`` is used by tests: pass ``":memory:"``.
    """

    def __init__(self, db_path: str | Path = "memory.db", alias_learning: bool = True) -> None:
        self._path = str(db_path)
        self._alias_learning = alias_learning
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- public API ----------------------------------------------------------

    def resolve_alias(self, variant: str) -> str | None:
        """Return the canonical form of a label variant, if learned."""
        row = self._conn.execute(
            "SELECT canonical FROM aliases WHERE variant = ?", (variant,)
        ).fetchone()
        if row is None:
            return None
        with self._lock:
            self._conn.execute(
                "UPDATE aliases SET hits = hits + 1 WHERE variant = ?", (variant,)
            )
            self._conn.commit()
        return str(row["canonical"])

    def learn_alias(self, variant: str, canonical: str) -> None:
        """Remember (or refresh) that ``variant`` means ``canonical``."""
        variant, canonical = variant.strip(), canonical.strip()
        if not variant or not canonical:
            return
        if not self._alias_learning:
            return
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO aliases (variant, canonical) VALUES (?, ?)
                ON CONFLICT(variant) DO UPDATE SET canonical = excluded.canonical,
                    updated_at = datetime('now')
                """,
                (variant, canonical),
            )
            self._conn.commit()
        logger.debug("memory learned alias '{}' -> '{}'", variant, canonical)

    def all_aliases(self) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute("SELECT variant, canonical FROM aliases").fetchall()
        return {str(r["variant"]): str(r["canonical"]) for r in rows}

    def forget_alias(self, variant: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM aliases WHERE variant = ?", (variant,))
            self._conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._conn.commit()

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row is not None else default

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except sqlite3.Error:
                pass

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["MemoryStore"]
