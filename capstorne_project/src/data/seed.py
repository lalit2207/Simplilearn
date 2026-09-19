"""Demo data for the assistant.

The seed deliberately mirrors the capstone's sample scenario - a 70-year-old
father with chronic kidney disease whose attendant wants a nephrologist - so
that scenario is runnable end to end without any manual data entry.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from src.data.database import connect, init_db, is_seeded, table_counts

# Working hours offered for every doctor, for the next SLOT_DAYS days.
SLOT_TIMES = ("09:00", "09:30", "10:00", "11:00", "11:30", "14:00", "15:00", "16:00")
SLOT_DAYS = 14

USERS = [
    ("Lalit Chauhan", "lalit@example.com", "+91-98100-11111"),
    ("Priya Sharma", "priya@example.com", "+91-98100-22222"),
]

# (attendant_index, relation, name, age, gender, blood_group, allergies)
PATIENTS = [
    (0, "father", "Ramesh Chauhan", 70, "Male", "B+", "Penicillin"),
    (0, "self", "Lalit Chauhan", 38, "Male", "O+", "None"),
    (0, "mother", "Sunita Chauhan", 66, "Female", "A+", "Sulfa drugs"),
    (1, "self", "Priya Sharma", 32, "Female", "AB+", "Peanuts"),
    (1, "child", "Aarav Sharma", 7, "Male", "O-", "None"),
]

# (patient_index, days_ago, record_type, condition, diagnosis, treatment,
#  medications, notes, alert_level)
HISTORY = [
    (
        0, 900, "diagnosis", "Chronic Kidney Disease",
        "CKD stage 3a diagnosed after persistently raised creatinine",
        "Low-protein renal diet, blood pressure control",
        "Telmisartan 40mg once daily",
        "Attendant reports fatigue and mild ankle swelling over three months. "
        "Nephrology referral advised at diagnosis.",
        "high",
    ),
    (
        0, 540, "diagnosis", "Type 2 Diabetes Mellitus",
        "Long-standing T2DM, HbA1c 8.1% - a major driver of his kidney decline",
        "Glycaemic control and quarterly renal monitoring",
        "Metformin 500mg twice daily",
        "Dose kept low because of reduced kidney function.",
        "medium",
    ),
    (
        0, 240, "lab_result", "Chronic Kidney Disease",
        "eGFR 42 mL/min/1.73m2, serum creatinine 1.9 mg/dL",
        "Continue conservative management, avoid NSAIDs",
        None,
        "Progression from stage 3a towards 3b. Repeat panel in three months.",
        "high",
    ),
    (
        0, 95, "note", "Hypertension",
        "Home readings averaging 148/88 mmHg",
        "Add low-dose diuretic, reduce dietary sodium",
        "Telmisartan 40mg, Chlorthalidone 6.25mg",
        "Family cooks separately for him now to keep salt intake down.",
        "medium",
    ),
    (
        0, 20, "procedure", "Chronic Kidney Disease",
        "Renal ultrasound: bilateral cortical thinning, no obstruction",
        "No intervention needed, continue medical management",
        None,
        "Attendant asking about newer treatment options and whether dialysis is near.",
        "medium",
    ),
    (
        2, 400, "diagnosis", "Osteoarthritis",
        "Bilateral knee osteoarthritis, moderate",
        "Physiotherapy and weight management",
        "Paracetamol as needed",
        "Avoids stairs. Declined steroid injection.",
        "low",
    ),
    (
        2, 60, "lab_result", "Hypothyroidism",
        "TSH 6.8 mIU/L, mildly elevated",
        "Start low-dose levothyroxine, recheck in eight weeks",
        "Levothyroxine 25mcg once daily",
        None,
        "low",
    ),
    (
        1, 150, "note", "Routine health check",
        "Vitals normal, borderline cholesterol",
        "Lifestyle advice only",
        None,
        "Family history of diabetes and kidney disease noted for screening.",
        "none",
    ),
    (
        4, 30, "diagnosis", "Allergic Rhinitis",
        "Seasonal allergic rhinitis",
        "Antihistamine during pollen season",
        "Cetirizine 5mg at night",
        "Symptoms worse on school sports days.",
        "low",
    ),
]

# (name, specialty, hospital, qualification, experience, rating, fee)
DOCTORS = [
    ("Dr. Anil Mehta", "Nephrologist", "Apollo Hospital, Delhi", "MD, DM (Nephrology)", 18, 4.8, 1200),
    ("Dr. Kavita Rao", "Nephrologist", "Fortis Healthcare, Gurgaon", "MD, DNB (Nephrology)", 12, 4.6, 1000),
    ("Dr. Suresh Iyer", "Nephrologist", "AIIMS, Delhi", "MD, DM (Nephrology)", 22, 4.9, 800),
    ("Dr. Neha Gupta", "Endocrinologist", "Max Healthcare, Saket", "MD, DM (Endocrinology)", 14, 4.7, 1100),
    ("Dr. Rajesh Khanna", "Cardiologist", "Medanta, Gurgaon", "MD, DM (Cardiology)", 20, 4.8, 1500),
    ("Dr. Meera Nair", "General Physician", "City Care Clinic, Noida", "MBBS, MD (Medicine)", 10, 4.5, 600),
    ("Dr. Vikram Bose", "Orthopedic Surgeon", "Fortis Healthcare, Noida", "MS (Orthopaedics)", 16, 4.6, 900),
    ("Dr. Ananya Sen", "Pediatrician", "Rainbow Children's Hospital", "MD (Pediatrics)", 11, 4.7, 700),
]


def _iso_days_ago(days: int) -> str:
    return (date.today() - timedelta(days=days)).isoformat()


def _generate_slots(doctor_count: int) -> list[tuple[int, str, str]]:
    """Build the doctor calendar: weekday slots only, starting tomorrow.

    Each doctor gets a rotating subset of times so availability differs
    between doctors and the agent has a real scheduling choice to make.
    """
    rows: list[tuple[int, str, str]] = []
    for doctor_id in range(1, doctor_count + 1):
        for offset in range(1, SLOT_DAYS + 1):
            day = date.today() + timedelta(days=offset)
            if day.weekday() >= 5:  # skip Saturday and Sunday
                continue
            stride = (doctor_id + offset) % 3 + 1
            for slot_time in SLOT_TIMES[:: stride]:
                rows.append((doctor_id, day.isoformat(), slot_time))
    return rows


def seed_database(db_path: Path | None = None, reset: bool = False) -> dict[str, int]:
    """Populate reference and demo data. Safe to call repeatedly."""
    init_db(db_path, reset=reset)

    if is_seeded(db_path) and not reset:
        return table_counts(db_path)

    with connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO users (full_name, email, phone) VALUES (?, ?, ?)",
            USERS,
        )
        conn.executemany(
            """INSERT INTO patients
                   (attendant_id, relation, full_name, age, gender, blood_group, allergies)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (attendant_index + 1, relation, name, age, gender, blood, allergies)
                for attendant_index, relation, name, age, gender, blood, allergies in PATIENTS
            ],
        )
        conn.executemany(
            """INSERT INTO medical_history
                   (patient_id, record_date, record_type, condition, diagnosis,
                    treatment, medications, notes, alert_level)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    patient_index + 1,
                    _iso_days_ago(days_ago),
                    record_type,
                    condition,
                    diagnosis,
                    treatment,
                    medications,
                    notes,
                    alert_level,
                )
                for (
                    patient_index, days_ago, record_type, condition, diagnosis,
                    treatment, medications, notes, alert_level,
                ) in HISTORY
            ],
        )
        conn.executemany(
            """INSERT INTO doctors
                   (full_name, specialty, hospital, qualification,
                    experience_years, rating, consultation_fee)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            DOCTORS,
        )
        conn.executemany(
            "INSERT OR IGNORE INTO slots (doctor_id, slot_date, slot_time) VALUES (?, ?, ?)",
            _generate_slots(len(DOCTORS)),
        )

    return table_counts(db_path)
