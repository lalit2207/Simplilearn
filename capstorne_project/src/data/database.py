"""SQLite access layer for the patient/EHR and doctor-schedule stores.

The agent tools in Step 3 never touch `sqlite3` directly - they call the
helpers here, so transaction handling and row conversion stay in one place.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from src.config import settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open a connection that returns dict-like rows and commits on success.

    Foreign keys are off by default in SQLite and must be enabled per
    connection, otherwise the ON DELETE CASCADE rules never fire.
    """
    path = db_path or settings.ehr_db_path
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | None = None, reset: bool = False) -> Path:
    """Create the schema. With `reset=True` the database file is recreated."""
    path = db_path or settings.ehr_db_path
    if reset and path.exists():
        path.unlink()

    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with connect(path) as conn:
        conn.executescript(schema)
    return path


def fetch_all(sql: str, params: Sequence[Any] = (), db_path: Path | None = None) -> list[dict]:
    with connect(db_path) as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def fetch_one(sql: str, params: Sequence[Any] = (), db_path: Path | None = None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None


def execute(sql: str, params: Sequence[Any] = (), db_path: Path | None = None) -> int:
    """Run a write statement and return the new row id (or rows affected)."""
    with connect(db_path) as conn:
        cursor = conn.execute(sql, params)
        return cursor.lastrowid if cursor.lastrowid else cursor.rowcount


def table_counts(db_path: Path | None = None) -> dict[str, int]:
    """Row count per table - used by the setup check and the Streamlit sidebar."""
    tables = ("users", "patients", "medical_history", "doctors", "slots", "appointments")
    with connect(db_path) as conn:
        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in tables
        }


def is_seeded(db_path: Path | None = None) -> bool:
    """True when reference data already exists, so seeding stays idempotent."""
    path = db_path or settings.ehr_db_path
    if not path.exists():
        return False
    with connect(path) as conn:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='doctors'"
        ).fetchone()
        if not exists:
            return False
        return conn.execute("SELECT COUNT(*) FROM doctors").fetchone()[0] > 0
