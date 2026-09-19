"""Patient and medical-record operations (the EHR side of the assistant).

These are plain Python functions returning plain data. The LangChain tool
wrappers live in `registry.py`, so the Streamlit UI can reuse the exact same
functions the agent calls without going through the LLM.
"""

from __future__ import annotations

from datetime import date

from src.data.database import execute, fetch_all, fetch_one

RELATION_ALIASES = {
    "me": "self",
    "myself": "self",
    "i": "self",
    "dad": "father",
    "papa": "father",
    "my father": "father",
    "mom": "mother",
    "mum": "mother",
    "mummy": "mother",
    "my mother": "mother",
    "wife": "spouse",
    "husband": "spouse",
    "son": "child",
    "daughter": "child",
    "kid": "child",
    "brother": "brother",
    "sister": "sister",
    "grandfather": "grandfather",
    "grandmother": "grandmother",
    "uncle": "uncle",
    "aunt": "aunt",
}

ALLOWED_RELATIONS = (
    "self",
    "father",
    "mother",
    "spouse",
    "child",
    "brother",
    "sister",
    "grandfather",
    "grandmother",
    "uncle",
    "aunt",
    "other",
)

ALERT_ORDER = {"high": 3, "medium": 2, "low": 1, "none": 0}


def list_patients(attendant_id: int | None = None) -> list[dict]:
    """All patients, optionally limited to one attendant's family."""
    if attendant_id is None:
        return fetch_all(
            "SELECT id, full_name, age, gender, relation, attendant_id FROM patients ORDER BY id"
        )
    return fetch_all(
        """SELECT id, full_name, age, gender, relation, attendant_id
           FROM patients WHERE attendant_id = ? ORDER BY id""",
        (attendant_id,),
    )


def search_patients(query: str, limit: int = 20) -> list[dict]:
    """Find patients by name, id, or allergy text for the doctor lookup page."""
    text = (query or "").strip()
    if not text:
        return []
    if text.isdigit():
        row = fetch_one(
            """SELECT id, full_name, age, gender, blood_group, allergies,
                      relation, attendant_id
               FROM patients WHERE id = ?""",
            (int(text),),
        )
        return [row] if row else []
    return fetch_all(
        """SELECT id, full_name, age, gender, blood_group, allergies,
                  relation, attendant_id
           FROM patients
           WHERE LOWER(full_name) LIKE ?
              OR LOWER(IFNULL(allergies, '')) LIKE ?
           ORDER BY full_name
           LIMIT ?""",
        (f"%{text.lower()}%", f"%{text.lower()}%", limit),
    )


def format_family_roster(attendant_id: int | None = None) -> str:
    """Readable list the agent can read out when it needs the attendant to choose."""
    members = list_patients(attendant_id)
    if not members:
        return "No family members are registered yet."
    return "Registered family members: " + "; ".join(
        f"{p['full_name']} (id {p['id']}, {p['relation']}, age {p.get('age') or '-'})"
        for p in members
    )


def match_patients(
    reference: str,
    attendant_id: int | None = None,
) -> list[dict]:
    """All patients that could match an id, relation, or name."""
    text = (reference or "").strip()
    if not text:
        return []

    if text.isdigit():
        row = fetch_one("SELECT * FROM patients WHERE id = ?", (int(text),))
        if row and attendant_id is not None and row.get("attendant_id") != attendant_id:
            return []
        return [row] if row else []

    key = text.lower().removeprefix("my ").strip()
    relation = RELATION_ALIASES.get(text.lower(), RELATION_ALIASES.get(key, key))
    found: dict[int, dict] = {}

    scope = "attendant_id = ?" if attendant_id is not None else "1=1"
    params: list = [attendant_id] if attendant_id is not None else []

    for row in fetch_all(
        f"SELECT * FROM patients WHERE {scope} AND LOWER(relation) = ?",
        (*params, relation),
    ):
        found[row["id"]] = row

    for row in fetch_all(
        f"SELECT * FROM patients WHERE {scope} AND LOWER(full_name) LIKE ?",
        (*params, f"%{text.lower()}%"),
    ):
        found[row["id"]] = row

    return list(found.values())


def resolve_patient(
    reference: str,
    attendant_id: int | None = None,
) -> dict | None:
    """Turn a loose reference into a patient row when the match is unique."""
    matches = match_patients(reference, attendant_id)
    if len(matches) == 1:
        return matches[0]
    return None


def register_attendant(
    full_name: str,
    email: str | None = None,
    phone: str | None = None,
) -> dict:
    """Create a caregiver account, then a self chart so they can book immediately."""
    name = (full_name or "").strip()
    if not name:
        return {"success": False, "error": "Enter your full name."}

    mail = (email or "").strip() or None
    number = (phone or "").strip() or None
    if mail:
        existing = fetch_one(
            "SELECT id FROM users WHERE LOWER(email) = ?", (mail.lower(),)
        )
        if existing:
            return {"success": False, "error": "That email is already registered. Sign in instead."}

    user_id = execute(
        "INSERT INTO users (full_name, email, phone) VALUES (?, ?, ?)",
        (name, mail, number),
    )
    register_patient(attendant_id=user_id, full_name=name, relation="self")
    return {
        "success": True,
        "attendant_id": user_id,
        "full_name": name,
        "email": mail,
    }


def register_patient(
    attendant_id: int,
    full_name: str,
    relation: str = "other",
    age: int | None = None,
    gender: str | None = None,
    blood_group: str | None = None,
    allergies: str | None = None,
) -> dict:
    """Register a family member so the attendant can keep their chart."""
    name = (full_name or "").strip()
    if not name:
        return {"success": False, "error": "A full name is required."}

    link = RELATION_ALIASES.get((relation or "").strip().lower(), (relation or "other").strip().lower())
    if link not in ALLOWED_RELATIONS:
        link = "other"

    existing = fetch_one(
        """SELECT id FROM patients
           WHERE attendant_id = ? AND LOWER(full_name) = ? AND LOWER(relation) = ?""",
        (attendant_id, name.lower(), link),
    )
    if existing:
        return {
            "success": False,
            "error": f"{name} ({link}) is already registered as patient id {existing['id']}.",
            "patient_id": existing["id"],
        }

    patient_id = execute(
        """INSERT INTO patients
               (attendant_id, relation, full_name, age, gender, blood_group, allergies)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (attendant_id, link, name, age, gender, blood_group, allergies),
    )
    return {"success": True, "patient_id": patient_id, "full_name": name, "relation": link}


def get_patient_profile(patient_id: int) -> dict | None:
    """Demographics plus the attendant's name."""
    return fetch_one(
        """SELECT p.*, u.full_name AS attendant_name
           FROM patients p
           LEFT JOIN users u ON u.id = p.attendant_id
           WHERE p.id = ?""",
        (patient_id,),
    )


def get_medical_history(patient_id: int, limit: int = 20) -> list[dict]:
    """Most recent records first - the order an LLM should summarise in."""
    return fetch_all(
        """SELECT id, record_date, record_type, condition, diagnosis,
                  treatment, medications, notes, alert_level
           FROM medical_history
           WHERE patient_id = ?
           ORDER BY record_date DESC, id DESC
           LIMIT ?""",
        (patient_id, limit),
    )


def get_active_alerts(patient_id: int) -> list[dict]:
    """Medium/high severity records, worst first - surfaced before booking."""
    records = fetch_all(
        """SELECT record_date, condition, diagnosis, alert_level
           FROM medical_history
           WHERE patient_id = ? AND alert_level IN ('medium', 'high')""",
        (patient_id,),
    )
    return sorted(
        records,
        key=lambda r: (ALERT_ORDER.get(r["alert_level"], 0), r["record_date"]),
        reverse=True,
    )


def add_medical_record(
    patient_id: int,
    condition: str | None = None,
    diagnosis: str | None = None,
    treatment: str | None = None,
    medications: str | None = None,
    notes: str | None = None,
    record_type: str = "note",
    alert_level: str = "none",
    record_date: str | None = None,
) -> dict:
    """Append a record. Supports both structured fields and free-text notes."""
    if not get_patient_profile(patient_id):
        return {"success": False, "error": f"No patient with id {patient_id}"}

    if alert_level not in ALERT_ORDER:
        return {
            "success": False,
            "error": f"alert_level must be one of {sorted(ALERT_ORDER)}",
        }

    record_id = execute(
        """INSERT INTO medical_history
               (patient_id, record_date, record_type, condition, diagnosis,
                treatment, medications, notes, alert_level)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            patient_id,
            record_date or date.today().isoformat(),
            record_type,
            condition,
            diagnosis,
            treatment,
            medications,
            notes,
            alert_level,
        ),
    )
    return {"success": True, "record_id": record_id, "patient_id": patient_id}


def update_medical_record(record_id: int, **fields) -> dict:
    """Patch named columns on an existing record."""
    allowed = {
        "record_date", "record_type", "condition", "diagnosis",
        "treatment", "medications", "notes", "alert_level",
    }
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return {"success": False, "error": f"Nothing to update. Allowed: {sorted(allowed)}"}

    if not fetch_one("SELECT id FROM medical_history WHERE id = ?", (record_id,)):
        return {"success": False, "error": f"No record with id {record_id}"}

    assignments = ", ".join(f"{column} = ?" for column in updates)
    execute(
        f"UPDATE medical_history SET {assignments} WHERE id = ?",
        (*updates.values(), record_id),
    )
    return {"success": True, "record_id": record_id, "updated": sorted(updates)}


def build_history_digest(patient_id: int, limit: int = 20) -> str:
    """Flatten a patient's chart into text for prompts and FAISS embedding.

    Everything the LLM needs about this patient becomes one readable block,
    which keeps the summarisation prompt in Step 5 simple.
    """
    profile = get_patient_profile(patient_id)
    if not profile:
        return f"No patient found with id {patient_id}."

    records = get_medical_history(patient_id, limit=limit)
    lines = [
        f"Patient: {profile['full_name']} (id {profile['id']})",
        f"Age: {profile.get('age')} | Gender: {profile.get('gender')} "
        f"| Blood group: {profile.get('blood_group')}",
        f"Known allergies: {profile.get('allergies') or 'None recorded'}",
        f"Relation to attendant: {profile.get('relation')} "
        f"({profile.get('attendant_name') or 'unknown attendant'})",
        "",
        f"Medical records on file: {len(records)}",
    ]

    if not records:
        lines.append("No medical history recorded yet.")

    for record in records:
        lines.append(
            f"\n[{record['record_date']}] {record['record_type']} "
            f"(alert: {record['alert_level']})"
        )
        for label, key in (
            ("Condition", "condition"),
            ("Diagnosis", "diagnosis"),
            ("Treatment", "treatment"),
            ("Medications", "medications"),
            ("Notes", "notes"),
        ):
            if record.get(key):
                lines.append(f"  {label}: {record[key]}")

    return "\n".join(lines)
