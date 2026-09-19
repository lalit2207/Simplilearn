"""Doctor clinic tools: patient search and slot add / remove / block."""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.planner import doctor_heuristic_plan  # noqa: E402
from src.data.seed import seed_database  # noqa: E402
from src.tools import appointment_tools as appts  # noqa: E402
from src.tools.patient_tools import register_attendant, search_patients  # noqa: E402
from src.tools.registry import DOCTOR_TOOLS  # noqa: E402


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    import src.data.database as database

    isolated = replace(database.settings, ehr_db_path=tmp_path / "ehr.db")
    monkeypatch.setattr(database, "settings", isolated)
    seed_database()
    return isolated.ehr_db_path


def test_register_attendant_creates_self_chart(isolated_db):
    created = register_attendant("Neha Verma", email="neha@example.com", phone="9810012345")
    assert created["success"]
    again = register_attendant("Someone Else", email="neha@example.com")
    assert not again["success"]
    matches = search_patients("Neha Verma")
    assert matches and matches[0]["relation"] == "self"


def test_search_patients_by_name_and_allergy(isolated_db):
    by_name = search_patients("Sunita")
    assert len(by_name) == 1
    assert by_name[0]["full_name"] == "Sunita Chauhan"

    by_allergy = search_patients("sulfa")
    assert any(row["full_name"] == "Sunita Chauhan" for row in by_allergy)

    by_id = search_patients("1")
    assert by_id and by_id[0]["id"] == 1


def test_add_remove_and_block_slots(isolated_db):
    day = (date.today() + timedelta(days=21)).isoformat()
    added = appts.add_slot(1, day, "10:00")
    assert added["success"]
    slot_id = added["slot_id"]

    open_slots = appts.find_available_slots(doctor_id=1, on_or_after=day, limit=20)
    assert any(row["slot_id"] == slot_id for row in open_slots)

    blocked = appts.set_unavailable(1, day, "10:00")
    assert blocked["success"]
    open_after = appts.find_available_slots(doctor_id=1, on_or_after=day, limit=20)
    assert all(row["slot_id"] != slot_id for row in open_after)

    book = appts.book_appointment(1, slot_id, reason="should fail")
    assert not book["success"]
    assert "unavailable" in book["error"].lower()

    restored = appts.restore_slot(slot_id, 1)
    assert restored["success"]
    removed = appts.remove_slot(slot_id, 1)
    assert removed["success"]


def test_cannot_remove_booked_slot(isolated_db):
    day = (date.today() + timedelta(days=22)).isoformat()
    added = appts.add_slot(1, day, "11:00")
    booked = appts.book_appointment(2, added["slot_id"], reason="review")
    assert booked["success"]
    result = appts.remove_slot(added["slot_id"], 1)
    assert not result["success"]
    assert "booked" in result["error"].lower()


def test_doctor_tools_and_heuristic():
    names = {tool.name for tool in DOCTOR_TOOLS}
    assert names == {
        "search_patients",
        "get_patient_history",
        "add_patient_record",
        "list_doctor_appointments",
        "search_medical_information",
    }
    plan = doctor_heuristic_plan("Summarise Ramesh Chauhan's medical history")
    assert "search_patients" in plan.tools_planned
    assert "get_patient_history" in plan.tools_planned
