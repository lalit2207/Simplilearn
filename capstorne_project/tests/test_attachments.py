"""Attachment extract, parse, and chart-filing without calling Groq."""

from __future__ import annotations

import io
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attachments.extract import extract_upload  # noqa: E402
from src.attachments.ingest import prepare_turn  # noqa: E402
from src.attachments.parse import extract_patient_name, is_medical_record, parse_record  # noqa: E402
from src.data.seed import seed_database  # noqa: E402
from src.tools.patient_tools import get_medical_history  # noqa: E402


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    import src.data.database as database

    isolated = replace(database.settings, ehr_db_path=tmp_path / "ehr.db")
    monkeypatch.setattr(database, "settings", isolated)
    seed_database()
    return isolated.ehr_db_path


def _docx_bytes(text: str) -> bytes:
    from docx import Document

    document = Document()
    document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


LAB_REPORT = """
Patient Name: Ramesh Chauhan
Diagnosis: Chronic kidney disease, stage 3
Medications: Losartan 50 mg daily
Lab result: Creatinine 1.8 mg/dL
"""


def test_extract_docx_and_patient_name():
    extracted = extract_upload("report.docx", _docx_bytes(LAB_REPORT))
    assert extracted.usable
    assert "Ramesh Chauhan" in extracted.text
    assert extract_patient_name(extracted.text) == "Ramesh Chauhan"
    assert is_medical_record(extracted.text)


def test_parse_record_fields():
    parsed = parse_record(LAB_REPORT)
    assert parsed.is_medical_record
    assert parsed.patient_name == "Ramesh Chauhan"
    assert parsed.diagnosis
    assert "kidney" in parsed.diagnosis.lower()


def test_plain_note_is_not_a_record():
    text = "Please book a visit next Tuesday if there is time."
    assert not is_medical_record(text)
    turn = prepare_turn(
        "What does this say?",
        [("note.docx", _docx_bytes(text), None)],
        role="attendant",
        attendant_id=1,
    )
    assert not turn.filed
    assert not turn.needs_name
    assert "What does this say?" in turn.agent_prompt


def test_files_named_record_on_matching_patient(isolated_db):
    turn = prepare_turn(
        "Please add this lab report",
        [("labs.docx", _docx_bytes(LAB_REPORT), None)],
        role="attendant",
        attendant_id=1,
    )
    assert turn.filed
    assert not turn.needs_name
    assert "already been added" in turn.agent_prompt
    history = get_medical_history(1)
    assert any("Imported from labs.docx" in (row.get("notes") or "") for row in history)


def test_asks_for_name_when_unclear(isolated_db):
    text = """
    Diagnosis: Hypertension
    Medications: Amlodipine 5 mg
    Treatment: Continue current plan
    """
    turn = prepare_turn(
        "",
        [("note.docx", _docx_bytes(text), None)],
        role="attendant",
        attendant_id=1,
    )
    assert turn.needs_name
    assert not turn.filed
    assert turn.pending_import
    assert "name is not clear" in turn.agent_prompt.lower() or "which patient" in turn.agent_prompt.lower()


def test_guest_cannot_file_a_record(isolated_db):
    before = get_medical_history(1)
    turn = prepare_turn(
        "File this",
        [("labs.docx", _docx_bytes(LAB_REPORT), None)],
        role="guest",
        attendant_id=None,
    )
    assert not turn.filed
    assert "sign in" in turn.agent_prompt.lower()
    assert len(get_medical_history(1)) == len(before)


def test_follow_up_name_files_waiting_record(isolated_db):
    first = prepare_turn(
        "",
        [("note.docx", _docx_bytes(
            "Diagnosis: Ankle swelling\nMedications: None\nTreatment: Elevate the leg\n"
        ), None)],
        role="attendant",
        attendant_id=1,
    )
    assert first.needs_name
    second = prepare_turn(
        "Ramesh Chauhan",
        [],
        role="attendant",
        attendant_id=1,
        pending_import=first.pending_import,
    )
    assert second.filed
    assert second.pending_import is None
