"""SQLite connection and transaction management helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "beauty_advisor.db"


def create_connection(database_path: str | Path = DEFAULT_DATABASE_PATH) -> sqlite3.Connection:
    """Open a SQLite connection configured for repository queries."""
    connection = sqlite3.connect(Path(database_path))
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def session_scope(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> Iterator[sqlite3.Connection]:
    """Yield a connection, committing successful work and closing it reliably."""
    connection = create_connection(database_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
