"""Doctor discovery and appointment booking (the scheduling side).

Booking is the one place where the agent mutates shared state, so the
write path is deliberately defensive: the slot claim and the appointment
insert happen inside a single transaction.
"""

from __future__ import annotations

from datetime import date

from src.data.database import connect, fetch_all, fetch_one

STANDARD_SLOT_TIMES = ("09:00", "09:30", "10:00", "11:00", "11:30", "14:00", "15:00", "16:00")
SLOT_STATUSES = ("available", "booked", "unavailable")

# Maps everyday phrasing onto the specialty values stored in `doctors`.
SPECIALTY_ALIASES = {
    "kidney": "Nephrologist",
    "kidney specialist": "Nephrologist",
    "renal": "Nephrologist",
    "nephrology": "Nephrologist",
    "nephrologist": "Nephrologist",
    "heart": "Cardiologist",
    "cardiology": "Cardiologist",
    "cardiologist": "Cardiologist",
    "diabetes": "Endocrinologist",
    "thyroid": "Endocrinologist",
    "hormone": "Endocrinologist",
    "endocrinology": "Endocrinologist",
    "endocrinologist": "Endocrinologist",
    "bone": "Orthopedic Surgeon",
    "joint": "Orthopedic Surgeon",
    "knee": "Orthopedic Surgeon",
    "orthopedic": "Orthopedic Surgeon",
    "orthopaedic": "Orthopedic Surgeon",
    "child": "Pediatrician",
    "children": "Pediatrician",
    "paediatrician": "Pediatrician",
    "pediatrician": "Pediatrician",
    "general": "General Physician",
    "physician": "General Physician",
    "gp": "General Physician",
}


def normalise_specialty(text: str) -> str:
    """Best-effort mapping of free text to a stored specialty name."""
    key = (text or "").strip().lower()
    if not key:
        return ""
    if key in SPECIALTY_ALIASES:
        return SPECIALTY_ALIASES[key]
    for alias, specialty in SPECIALTY_ALIASES.items():
        if alias in key:
            return specialty
    return text.strip().title()


def list_specialties() -> list[str]:
    rows = fetch_all("SELECT DISTINCT specialty FROM doctors ORDER BY specialty")
    return [row["specialty"] for row in rows]


def find_doctors(specialty: str | None = None, limit: int = 10) -> list[dict]:
    """Doctors for a specialty, best-rated first."""
    if specialty:
        return fetch_all(
            """SELECT id, full_name, specialty, hospital, qualification,
                      experience_years, rating, consultation_fee
               FROM doctors
               WHERE LOWER(specialty) = LOWER(?)
               ORDER BY rating DESC, experience_years DESC
               LIMIT ?""",
            (normalise_specialty(specialty), limit),
        )
    return fetch_all(
        """SELECT id, full_name, specialty, hospital, qualification,
                  experience_years, rating, consultation_fee
           FROM doctors ORDER BY specialty, rating DESC LIMIT ?""",
        (limit,),
    )


def find_available_slots(
    specialty: str | None = None,
    doctor_id: int | None = None,
    on_or_after: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Earliest open slots, filtered by specialty and/or a specific doctor."""
    clauses = ["s.status = 'available'", "s.slot_date >= ?"]
    params: list = [on_or_after or date.today().isoformat()]

    if doctor_id is not None:
        clauses.append("s.doctor_id = ?")
        params.append(doctor_id)
    if specialty:
        clauses.append("LOWER(d.specialty) = LOWER(?)")
        params.append(normalise_specialty(specialty))

    params.append(limit)
    return fetch_all(
        f"""SELECT s.id AS slot_id, s.slot_date, s.slot_time,
                   d.id AS doctor_id, d.full_name AS doctor_name,
                   d.specialty, d.hospital, d.rating, d.consultation_fee
            FROM slots s
            JOIN doctors d ON d.id = s.doctor_id
            WHERE {' AND '.join(clauses)}
            ORDER BY s.slot_date, s.slot_time, d.rating DESC
            LIMIT ?""",
        params,
    )


def book_appointment(
    patient_id: int,
    slot_id: int,
    reason: str | None = None,
) -> dict:
    """Claim a slot and create the appointment atomically.

    The `AND status = 'available'` guard is what makes this safe: if another
    booking took the slot first, the UPDATE affects zero rows and we abort
    before inserting an appointment.
    """
    with connect() as conn:
        patient = conn.execute(
            "SELECT id, full_name FROM patients WHERE id = ?", (patient_id,)
        ).fetchone()
        if not patient:
            return {"success": False, "error": f"No patient with id {patient_id}"}

        slot = conn.execute(
            """SELECT s.id, s.doctor_id, s.slot_date, s.slot_time, s.status,
                      d.full_name AS doctor_name, d.specialty, d.hospital
               FROM slots s JOIN doctors d ON d.id = s.doctor_id
               WHERE s.id = ?""",
            (slot_id,),
        ).fetchone()
        if not slot:
            return {"success": False, "error": f"No slot with id {slot_id}"}
        if slot["status"] == "unavailable":
            return {
                "success": False,
                "error": f"Slot {slot_id} is blocked as unavailable and cannot be booked",
            }
        if slot["status"] != "available":
            return {"success": False, "error": f"Slot {slot_id} is already booked"}

        claimed = conn.execute(
            "UPDATE slots SET status = 'booked' WHERE id = ? AND status = 'available'",
            (slot_id,),
        )
        if claimed.rowcount != 1:
            return {"success": False, "error": f"Slot {slot_id} was just taken"}

        cursor = conn.execute(
            """INSERT INTO appointments (patient_id, doctor_id, slot_id, reason)
               VALUES (?, ?, ?, ?)""",
            (patient_id, slot["doctor_id"], slot_id, reason),
        )

        return {
            "success": True,
            "appointment_id": cursor.lastrowid,
            "patient_name": patient["full_name"],
            "doctor_name": slot["doctor_name"],
            "specialty": slot["specialty"],
            "hospital": slot["hospital"],
            "slot_date": slot["slot_date"],
            "slot_time": slot["slot_time"],
            "reason": reason,
        }


def cancel_appointment(appointment_id: int) -> dict:
    """Cancel an appointment and release its slot back to the calendar."""
    with connect() as conn:
        appointment = conn.execute(
            "SELECT id, slot_id, status FROM appointments WHERE id = ?",
            (appointment_id,),
        ).fetchone()
        if not appointment:
            return {"success": False, "error": f"No appointment with id {appointment_id}"}
        if appointment["status"] == "cancelled":
            return {"success": False, "error": f"Appointment {appointment_id} is already cancelled"}

        conn.execute(
            "UPDATE appointments SET status = 'cancelled' WHERE id = ?", (appointment_id,)
        )
        conn.execute(
            "UPDATE slots SET status = 'available' WHERE id = ?", (appointment["slot_id"],)
        )
        return {"success": True, "appointment_id": appointment_id, "status": "cancelled"}


def list_appointments(
    patient_id: int | None = None,
    doctor_id: int | None = None,
    status: str | None = None,
) -> list[dict]:
    """Read from the flattened view - powers the Streamlit tracker."""
    clauses: list[str] = []
    params: list = []
    if patient_id is not None:
        clauses.append("patient_id = ?")
        params.append(patient_id)
    if doctor_id is not None:
        clauses.append("doctor_id = ?")
        params.append(doctor_id)
    if status:
        clauses.append("status = ?")
        params.append(status)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return fetch_all(
        f"SELECT * FROM appointment_details {where} ORDER BY slot_date, slot_time",
        params,
    )


def get_appointment(appointment_id: int) -> dict | None:
    return fetch_one(
        "SELECT * FROM appointment_details WHERE appointment_id = ?", (appointment_id,)
    )


def list_doctor_slots(
    doctor_id: int,
    slot_date: str | None = None,
    status: str | None = None,
    on_or_after: str | None = None,
) -> list[dict]:
    """The signed-in doctor's calendar, including blocked times."""
    clauses = ["doctor_id = ?"]
    params: list = [doctor_id]
    if slot_date:
        clauses.append("slot_date = ?")
        params.append(slot_date)
    elif on_or_after:
        clauses.append("slot_date >= ?")
        params.append(on_or_after)
    if status:
        clauses.append("status = ?")
        params.append(status)
    return fetch_all(
        f"""SELECT id AS slot_id, doctor_id, slot_date, slot_time, status
            FROM slots
            WHERE {' AND '.join(clauses)}
            ORDER BY slot_date, slot_time""",
        params,
    )


def add_slot(doctor_id: int, slot_date: str, slot_time: str) -> dict:
    """Open a bookable slot. Restores a blocked time; never overwrites a booking."""
    if not fetch_one("SELECT id FROM doctors WHERE id = ?", (doctor_id,)):
        return {"success": False, "error": f"No doctor with id {doctor_id}"}

    existing = fetch_one(
        """SELECT id, status FROM slots
           WHERE doctor_id = ? AND slot_date = ? AND slot_time = ?""",
        (doctor_id, slot_date, slot_time),
    )
    if existing:
        if existing["status"] == "booked":
            return {
                "success": False,
                "error": f"{slot_date} {slot_time} is already booked.",
                "slot_id": existing["id"],
            }
        if existing["status"] == "unavailable":
            with connect() as conn:
                conn.execute(
                    "UPDATE slots SET status = 'available' WHERE id = ?",
                    (existing["id"],),
                )
            return {
                "success": True,
                "slot_id": existing["id"],
                "slot_date": slot_date,
                "slot_time": slot_time,
                "restored": True,
            }
        return {
            "success": True,
            "slot_id": existing["id"],
            "slot_date": slot_date,
            "slot_time": slot_time,
            "already_open": True,
        }

    with connect() as conn:
        cursor = conn.execute(
            """INSERT INTO slots (doctor_id, slot_date, slot_time, status)
               VALUES (?, ?, ?, 'available')""",
            (doctor_id, slot_date, slot_time),
        )
        return {
            "success": True,
            "slot_id": cursor.lastrowid,
            "slot_date": slot_date,
            "slot_time": slot_time,
        }


def remove_slot(slot_id: int, doctor_id: int | None = None) -> dict:
    """Delete an open or blocked slot. Booked times stay until cancelled."""
    clauses = ["id = ?"]
    params: list = [slot_id]
    if doctor_id is not None:
        clauses.append("doctor_id = ?")
        params.append(doctor_id)
    slot = fetch_one(
        f"SELECT id, status, slot_date, slot_time FROM slots WHERE {' AND '.join(clauses)}",
        params,
    )
    if not slot:
        return {"success": False, "error": f"No slot with id {slot_id}"}
    if slot["status"] == "booked":
        return {"success": False, "error": "A booked slot cannot be removed. Cancel the visit first."}
    with connect() as conn:
        conn.execute("DELETE FROM slots WHERE id = ?", (slot["id"],))
    return {
        "success": True,
        "slot_id": slot["id"],
        "slot_date": slot["slot_date"],
        "slot_time": slot["slot_time"],
    }


def set_unavailable(doctor_id: int, slot_date: str, slot_time: str) -> dict:
    """Block a time so attendants cannot book it."""
    if not fetch_one("SELECT id FROM doctors WHERE id = ?", (doctor_id,)):
        return {"success": False, "error": f"No doctor with id {doctor_id}"}

    existing = fetch_one(
        """SELECT id, status FROM slots
           WHERE doctor_id = ? AND slot_date = ? AND slot_time = ?""",
        (doctor_id, slot_date, slot_time),
    )
    if existing:
        if existing["status"] == "booked":
            return {
                "success": False,
                "error": f"{slot_date} {slot_time} is booked. Cancel the visit before blocking it.",
                "slot_id": existing["id"],
            }
        with connect() as conn:
            conn.execute(
                "UPDATE slots SET status = 'unavailable' WHERE id = ?",
                (existing["id"],),
            )
        return {
            "success": True,
            "slot_id": existing["id"],
            "slot_date": slot_date,
            "slot_time": slot_time,
        }

    with connect() as conn:
        cursor = conn.execute(
            """INSERT INTO slots (doctor_id, slot_date, slot_time, status)
               VALUES (?, ?, ?, 'unavailable')""",
            (doctor_id, slot_date, slot_time),
        )
        return {
            "success": True,
            "slot_id": cursor.lastrowid,
            "slot_date": slot_date,
            "slot_time": slot_time,
        }


def restore_slot(slot_id: int, doctor_id: int | None = None) -> dict:
    """Turn a blocked time back into an open slot."""
    clauses = ["id = ?"]
    params: list = [slot_id]
    if doctor_id is not None:
        clauses.append("doctor_id = ?")
        params.append(doctor_id)
    slot = fetch_one(
        f"SELECT id, status, slot_date, slot_time FROM slots WHERE {' AND '.join(clauses)}",
        params,
    )
    if not slot:
        return {"success": False, "error": f"No slot with id {slot_id}"}
    if slot["status"] != "unavailable":
        return {"success": False, "error": "Only a blocked slot can be restored."}
    with connect() as conn:
        conn.execute("UPDATE slots SET status = 'available' WHERE id = ?", (slot["id"],))
    return {
        "success": True,
        "slot_id": slot["id"],
        "slot_date": slot["slot_date"],
        "slot_time": slot["slot_time"],
    }


def format_doctor_roster(doctor_id: int) -> str:
    """Patients currently booked with this doctor, for the doctor-agent prompt."""
    rows = list_appointments(doctor_id=doctor_id, status="confirmed")
    if not rows:
        return (
            "No confirmed appointments on your list yet. "
            "Search any registered patient by name or id."
        )
    unique: dict[int, str] = {}
    for row in rows:
        unique[row["patient_id"]] = row["patient_name"]
    return "Patients booked with you: " + "; ".join(
        f"{name} (id {patient_id})" for patient_id, name in unique.items()
    )
