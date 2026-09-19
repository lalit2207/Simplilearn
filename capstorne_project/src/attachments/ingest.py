"""Turn attachments plus a typed question into a chat turn the agent can run."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.attachments.extract import ExtractedFile, extract_upload
from src.attachments.parse import ParsedRecord, looks_like_name_reply, parse_record, wants_to_file
from src.tools.patient_tools import add_medical_record, match_patients


@dataclass
class PreparedTurn:
    display: str
    agent_prompt: str
    attachments: list[str] = field(default_factory=list)
    filed: bool = False
    needs_name: bool = False
    pending_import: dict | None = None


def prepare_turn(
    question: str,
    files: list[tuple[str, bytes, str | None]],
    *,
    role: str,
    attendant_id: int | None,
    pending_import: dict | None = None,
) -> PreparedTurn:
    """Extract files, file a chart when the patient is unique, else ask for the name."""
    extracts = [extract_upload(name, data, mime) for name, data, mime in files]
    names = [item.name for item in extracts]
    typed = (question or "").strip()

    if not extracts and pending_import:
        return _resume_pending(typed, pending_import, role, attendant_id)

    display = _display(typed, names)
    excerpt = _excerpt(extracts)
    if not extracts:
        return PreparedTurn(display=typed, agent_prompt=typed)

    combined = "\n\n".join(item.text for item in extracts if item.text)
    parsed = parse_record(combined, typed)
    medical = parsed.is_medical_record or wants_to_file(typed)

    if not medical:
        return PreparedTurn(
            display=display,
            agent_prompt=_question_prompt(typed, excerpt, role),
            attachments=names,
            pending_import=pending_import,
        )

    if role == "guest":
        return PreparedTurn(
            display=display,
            agent_prompt=_guest_record_prompt(typed, excerpt),
            attachments=names,
            needs_name=False,
        )

    matches = _resolve_patient(typed, parsed.patient_name, attendant_id if role == "attendant" else None)
    if len(matches) == 1:
        filed = _file_record(matches[0]["id"], parsed, names)
        label = f"{matches[0]['full_name']} (id {matches[0]['id']})"
        return PreparedTurn(
            display=display,
            agent_prompt=_filed_prompt(typed, excerpt, label, filed),
            attachments=names,
            filed=bool(filed.get("success")),
        )

    waiting = _pending_payload(parsed, excerpt, names)
    if len(matches) > 1:
        options = "; ".join(f"{row['full_name']} (id {row['id']})" for row in matches)
        return PreparedTurn(
            display=display,
            agent_prompt=_ask_which_prompt(typed, excerpt, options),
            attachments=names,
            needs_name=True,
            pending_import=waiting,
        )

    return PreparedTurn(
        display=display,
        agent_prompt=_ask_name_prompt(typed, excerpt, parsed.patient_name),
        attachments=names,
        needs_name=True,
        pending_import=waiting,
    )


def _resume_pending(
    typed: str,
    pending: dict,
    role: str,
    attendant_id: int | None,
) -> PreparedTurn:
    if role == "guest":
        return PreparedTurn(
            display=typed,
            agent_prompt=typed,
            pending_import=pending,
        )
    if not looks_like_name_reply(typed):
        return PreparedTurn(display=typed, agent_prompt=typed, pending_import=pending)

    parsed = ParsedRecord(
        is_medical_record=True,
        patient_name=pending.get("patient_name"),
        condition=pending.get("condition"),
        diagnosis=pending.get("diagnosis"),
        treatment=pending.get("treatment"),
        medications=pending.get("medications"),
        notes=pending.get("notes"),
        record_type=pending.get("record_type") or "note",
        alert_level=pending.get("alert_level") or "none",
    )
    matches = _resolve_patient(typed, parsed.patient_name, attendant_id if role == "attendant" else None)
    excerpt = pending.get("excerpt") or pending.get("notes") or ""
    names = pending.get("filenames") or []

    if len(matches) == 1:
        filed = _file_record(matches[0]["id"], parsed, names)
        label = f"{matches[0]['full_name']} (id {matches[0]['id']})"
        return PreparedTurn(
            display=typed,
            agent_prompt=_filed_prompt(typed, excerpt, label, filed),
            attachments=names,
            filed=bool(filed.get("success")),
        )
    if len(matches) > 1:
        options = "; ".join(f"{row['full_name']} (id {row['id']})" for row in matches)
        return PreparedTurn(
            display=typed,
            agent_prompt=_ask_which_prompt(typed, excerpt, options),
            attachments=names,
            needs_name=True,
            pending_import=pending,
        )
    return PreparedTurn(
        display=typed,
        agent_prompt=(
            f"{typed}\n\nThe user named a patient for the waiting attachment, but no "
            "matching chart was found. Ask them to confirm the full name or register "
            "the person first. Do not invent a patient.\n\n"
            f"{excerpt}"
        ),
        attachments=names,
        needs_name=True,
        pending_import=pending,
    )


def _resolve_patient(
    question: str,
    document_name: str | None,
    attendant_id: int | None,
) -> list[dict]:
    for reference in _name_references(question, document_name):
        matches = match_patients(reference, attendant_id)
        if matches:
            return matches
    return []


def _name_references(question: str, document_name: str | None) -> list[str]:
    refs: list[str] = []
    for alias in (
        "my father",
        "my mother",
        "my wife",
        "my husband",
        "my son",
        "my daughter",
        "my brother",
        "my sister",
    ):
        if alias in (question or "").lower():
            refs.append(alias)
    named = re.search(
        r"\b(?:for|on|to)\s+([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,2})",
        question or "",
    )
    if named:
        refs.append(named.group(1).strip())
    if document_name:
        refs.append(document_name)
    if question:
        refs.append(question)
    return refs


def _file_record(patient_id: int, parsed: ParsedRecord, filenames: list[str]) -> dict:
    source = ", ".join(filenames) if filenames else "uploaded file"
    notes = parsed.notes or ""
    if source not in notes:
        notes = f"Imported from {source}. {notes}".strip()
    return add_medical_record(
        patient_id,
        condition=parsed.condition,
        diagnosis=parsed.diagnosis,
        treatment=parsed.treatment,
        medications=parsed.medications,
        notes=notes[:1500],
        record_type=parsed.record_type or "note",
        alert_level=parsed.alert_level or "none",
    )


def _pending_payload(parsed: ParsedRecord, excerpt: str, names: list[str]) -> dict:
    return {
        "patient_name": parsed.patient_name,
        "condition": parsed.condition,
        "diagnosis": parsed.diagnosis,
        "treatment": parsed.treatment,
        "medications": parsed.medications,
        "notes": parsed.notes,
        "record_type": parsed.record_type,
        "alert_level": parsed.alert_level,
        "excerpt": excerpt,
        "filenames": names,
    }


def _display(typed: str, names: list[str]) -> str:
    if typed:
        return typed
    if names:
        return f"Shared {', '.join(names)}"
    return typed


def _excerpt(extracts: list[ExtractedFile]) -> str:
    blocks = []
    for item in extracts:
        if item.text:
            blocks.append(f"### {item.name}\n{item.text}")
        else:
            blocks.append(f"### {item.name}\n[Could not read file: {item.error}]")
    return "\n\n".join(blocks)


def _question_prompt(typed: str, excerpt: str, role: str) -> str:
    ask = typed or "Please review the attached file and tell me what it contains."
    guest = ""
    if role == "guest":
        guest = (
            "\nIf this later turns out to be a personal medical record, remind them "
            "that filing it on a chart needs a sign-in."
        )
    return (
        f"{ask}\n\nThe user attached the following file(s). Use this content to "
        f"answer. Do not invent details that are not in the file.{guest}\n\n{excerpt}"
    )


def _guest_record_prompt(typed: str, excerpt: str) -> str:
    ask = typed or "Please review this medical record."
    return (
        f"{ask}\n\nThe attached file looks like a medical record. Guests cannot add "
        "it to a patient chart. Summarise what you can read, then ask them to sign "
        "in as a registered user (account menu) so it can be filed. Do not invent "
        f"a patient or call chart tools.\n\n{excerpt}"
    )


def _filed_prompt(typed: str, excerpt: str, label: str, result: dict) -> str:
    ask = typed or "Please file this medical record."
    if result.get("success"):
        status = (
            f"The attachment has already been added to {label} as record "
            f"id {result.get('record_id')}. Confirm what was filed in plain language. "
            "Do not call add_patient_record again for the same document."
        )
    else:
        status = (
            f"Tried to file the attachment on {label} but it failed: "
            f"{result.get('error')}. Explain this and offer to retry."
        )
    return f"{ask}\n\n{status}\n\n{excerpt}"


def _ask_name_prompt(typed: str, excerpt: str, found_name: str | None) -> str:
    ask = typed or "Please file this medical record."
    hint = (
        f"A possible name on the document is {found_name}, but it did not match a chart."
        if found_name
        else "The patient name is not clear from the document."
    )
    return (
        f"{ask}\n\nThe attached file looks like a medical record. {hint} "
        "Ask which patient this belongs to (name or relation). Do not guess. "
        "Do not call add_patient_record until they name someone.\n\n"
        f"{excerpt}"
    )


def _ask_which_prompt(typed: str, excerpt: str, options: str) -> str:
    ask = typed or "Please file this medical record."
    return (
        f"{ask}\n\nThe attached file looks like a medical record. Several patients "
        f"could match: {options}. Ask which one. Do not guess.\n\n{excerpt}"
    )
