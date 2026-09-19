"""Saved assistant threads: registered users persist, owners stay isolated."""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.chat_store import (  # noqa: E402
    create_session,
    get_session,
    latest_session,
    list_sessions,
    load_messages,
    save_session,
    title_from_messages,
)
from src.data.seed import seed_database  # noqa: E402


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    import src.data.chat_store as chat_store
    import src.data.database as database

    isolated = replace(database.settings, ehr_db_path=tmp_path / "ehr.db")
    monkeypatch.setattr(database, "settings", isolated)
    monkeypatch.setattr(chat_store, "execute", database.execute)
    seed_database()
    return isolated.ehr_db_path


def test_title_from_first_user_message():
    assert title_from_messages([]) == "New assistant"
    assert title_from_messages([{"role": "assistant", "text": "Hi"}]) == "New assistant"
    assert (
        title_from_messages([{"role": "user", "text": "  Book a nephrologist  "}])
        == "Book a nephrologist"
    )
    long_text = "A" * 80
    title = title_from_messages([{"role": "user", "text": long_text}])
    assert title.endswith("…")
    assert len(title) == 48


def test_save_and_reload_session(isolated_db):
    session_id = create_session("attendant", 1, thread_id="thread-a")
    save_session(
        session_id,
        [
            {
                "role": "user",
                "text": "Summarise my father's history",
                "attachments": ["lab.pdf"],
                "sent_at": datetime(2026, 9, 19, 10, 0, 0),
            },
            {
                "role": "assistant",
                "text": "Ramesh has CKD stage 3a.",
                "sent_at": datetime(2026, 9, 19, 10, 0, 0),
                "received_at": datetime(2026, 9, 19, 10, 0, 8),
                "duration_s": 8.0,
            },
        ],
        thread_id="thread-a",
    )

    row = get_session(session_id)
    assert row["title"] == "Summarise my father's history"
    assert row["thread_id"] == "thread-a"

    messages = load_messages(session_id)
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[0]["attachments"] == ["lab.pdf"]
    assert messages[1]["text"].startswith("Ramesh")
    assert messages[1]["duration_s"] == 8.0
    assert messages[0]["sent_at"].hour == 10


def test_owners_are_isolated(isolated_db):
    first = create_session("attendant", 1)
    save_session(first, [{"role": "user", "text": "Lalit thread"}])
    second = create_session("attendant", 2)
    save_session(second, [{"role": "user", "text": "Priya thread"}])
    doctor = create_session("doctor", 1)
    save_session(doctor, [{"role": "user", "text": "Clinic thread"}])

    lalit = list_sessions("attendant", 1)
    assert [row["title"] for row in lalit] == ["Lalit thread"]
    assert latest_session("attendant", 2)["title"] == "Priya thread"
    assert latest_session("doctor", 1)["title"] == "Clinic thread"
    assert latest_session("attendant", 99) is None
