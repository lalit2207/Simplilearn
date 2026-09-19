"""Persist registered-user assistant threads in SQLite.

Guests never write here. The Streamlit UI holds the live transcript; this
module is the durable copy so a signed-in attendant or doctor can reopen it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.database import connect, execute, fetch_all, fetch_one

OWNER_ATTENDANT = "attendant"
OWNER_DOCTOR = "doctor"
DEFAULT_TITLE = "New assistant"
TITLE_LIMIT = 48


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    return str(value)


def _parse_dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _attachments_dumps(value: Any) -> str:
    if not value:
        return "[]"
    if isinstance(value, str):
        return value
    return json.dumps(list(value))


def _attachments_loads(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data]


def title_from_messages(messages: list[dict]) -> str:
    """First user line, trimmed, or the empty-session label."""
    for item in messages:
        role = item.get("role") if isinstance(item, dict) else item[0]
        text = item.get("text") if isinstance(item, dict) else item[1]
        if role != "user":
            continue
        line = " ".join(str(text or "").split())
        if not line:
            continue
        if len(line) <= TITLE_LIMIT:
            return line
        return line[: TITLE_LIMIT - 1] + "…"
    return DEFAULT_TITLE


def create_session(
    owner_type: str,
    owner_id: int,
    *,
    thread_id: str | None = None,
    title: str = DEFAULT_TITLE,
    db_path: Path | None = None,
) -> int:
    return execute(
        """INSERT INTO assistant_sessions (owner_type, owner_id, thread_id, title)
           VALUES (?, ?, ?, ?)""",
        (owner_type, owner_id, thread_id, title or DEFAULT_TITLE),
        db_path=db_path,
    )


def get_session(session_id: int, db_path: Path | None = None) -> dict | None:
    return fetch_one(
        "SELECT * FROM assistant_sessions WHERE id = ?",
        (session_id,),
        db_path=db_path,
    )


def list_sessions(
    owner_type: str,
    owner_id: int,
    *,
    limit: int = 12,
    db_path: Path | None = None,
) -> list[dict]:
    return fetch_all(
        """SELECT id, owner_type, owner_id, thread_id, title, created_at, updated_at
           FROM assistant_sessions
           WHERE owner_type = ? AND owner_id = ?
           ORDER BY datetime(updated_at) DESC, id DESC
           LIMIT ?""",
        (owner_type, owner_id, limit),
        db_path=db_path,
    )


def latest_session(
    owner_type: str, owner_id: int, db_path: Path | None = None
) -> dict | None:
    rows = list_sessions(owner_type, owner_id, limit=1, db_path=db_path)
    return rows[0] if rows else None


def load_messages(session_id: int, db_path: Path | None = None) -> list[dict]:
    rows = fetch_all(
        """SELECT role, text, attachments, sent_at, received_at, duration_s
           FROM assistant_messages
           WHERE session_id = ?
           ORDER BY sort_order ASC, id ASC""",
        (session_id,),
        db_path=db_path,
    )
    chat: list[dict] = []
    for row in rows:
        chat.append(
            {
                "role": row["role"],
                "text": row.get("text") or "",
                "attachments": _attachments_loads(row.get("attachments")),
                "sent_at": _parse_dt(row.get("sent_at")),
                "received_at": _parse_dt(row.get("received_at")),
                "duration_s": row.get("duration_s"),
                "trace": None,
            }
        )
    return chat


def save_session(
    session_id: int,
    messages: list[dict],
    *,
    title: str | None = None,
    thread_id: str | None = None,
    db_path: Path | None = None,
) -> None:
    label = title or title_from_messages(messages)
    with connect(db_path) as conn:
        fields = ["title = ?", "updated_at = datetime('now')"]
        params: list[Any] = [label]
        if thread_id:
            fields.insert(0, "thread_id = ?")
            params.insert(0, thread_id)
        params.append(session_id)
        conn.execute(
            f"UPDATE assistant_sessions SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        conn.execute(
            "DELETE FROM assistant_messages WHERE session_id = ?",
            (session_id,),
        )
        rows = []
        for index, item in enumerate(messages):
            if not isinstance(item, dict):
                role, text = item
                item = {"role": role, "text": text}
            rows.append(
                (
                    session_id,
                    index,
                    item.get("role") or "user",
                    item.get("text") or "",
                    _attachments_dumps(item.get("attachments")),
                    _iso(item.get("sent_at")),
                    _iso(item.get("received_at")),
                    item.get("duration_s"),
                )
            )
        if rows:
            conn.executemany(
                """INSERT INTO assistant_messages
                       (session_id, sort_order, role, text, attachments,
                        sent_at, received_at, duration_s)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
