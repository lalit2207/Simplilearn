-- ============================================================
-- Agentic Healthcare Assistant - relational schema
--
-- Two systems the capstone asks for live here:
--   * Patient / EHR DB   -> users, patients, medical_history
--   * Doctor Schedule API -> doctors, slots, appointments
-- ============================================================

PRAGMA foreign_keys = ON;

-- Attendants / caregivers who talk to the assistant. This table is what
-- lets the agent resolve phrases like "my father" into a real patient id.
CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name  TEXT NOT NULL,
    email      TEXT UNIQUE,
    phone      TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS patients (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    attendant_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    relation     TEXT NOT NULL DEFAULT 'self',   -- self, father, mother, spouse, child, sibling, other
    full_name    TEXT NOT NULL,
    age          INTEGER,
    gender       TEXT,
    blood_group  TEXT,
    allergies    TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Structured *and* unstructured history: the fixed columns cover the
-- structured side, while `notes` holds free text the attendant dictates.
CREATE TABLE IF NOT EXISTS medical_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id  INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    record_date TEXT NOT NULL,
    record_type TEXT NOT NULL DEFAULT 'diagnosis',  -- diagnosis | lab_result | medication | procedure | note
    condition   TEXT,
    diagnosis   TEXT,
    treatment   TEXT,
    medications TEXT,
    notes       TEXT,
    alert_level TEXT NOT NULL DEFAULT 'none',       -- none | low | medium | high
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS doctors (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name        TEXT NOT NULL,
    specialty        TEXT NOT NULL,
    hospital         TEXT,
    qualification    TEXT,
    experience_years INTEGER,
    rating           REAL,
    consultation_fee INTEGER
);

-- One row per bookable time slot. `status` is the lock that stops the
-- agent from double-booking the same doctor.
CREATE TABLE IF NOT EXISTS slots (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_id INTEGER NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,
    slot_date TEXT NOT NULL,                       -- YYYY-MM-DD
    slot_time TEXT NOT NULL,                       -- HH:MM
    status    TEXT NOT NULL DEFAULT 'available',   -- available | booked | unavailable
    UNIQUE (doctor_id, slot_date, slot_time)
);

CREATE TABLE IF NOT EXISTS appointments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    doctor_id  INTEGER NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,
    slot_id    INTEGER NOT NULL UNIQUE REFERENCES slots(id) ON DELETE CASCADE,
    reason     TEXT,
    status     TEXT NOT NULL DEFAULT 'confirmed',  -- confirmed | cancelled | completed
    booked_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Saved assistant threads. Guests never write here; registered attendants
-- and signed-in doctors reopen past sessions from the left menu.
CREATE TABLE IF NOT EXISTS assistant_sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_type  TEXT NOT NULL,                       -- attendant | doctor
    owner_id    INTEGER NOT NULL,
    thread_id   TEXT,
    title       TEXT NOT NULL DEFAULT 'New assistant',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS assistant_messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER NOT NULL REFERENCES assistant_sessions(id) ON DELETE CASCADE,
    sort_order   INTEGER NOT NULL,
    role         TEXT NOT NULL,                      -- user | assistant
    text         TEXT NOT NULL DEFAULT '',
    attachments  TEXT NOT NULL DEFAULT '[]',
    sent_at      TEXT,
    received_at  TEXT,
    duration_s   REAL
);

CREATE INDEX IF NOT EXISTS idx_history_patient  ON medical_history (patient_id, record_date DESC);
CREATE INDEX IF NOT EXISTS idx_slots_lookup     ON slots (doctor_id, slot_date, status);
CREATE INDEX IF NOT EXISTS idx_doctors_specialty ON doctors (specialty);
CREATE INDEX IF NOT EXISTS idx_appt_patient     ON appointments (patient_id, status);
CREATE INDEX IF NOT EXISTS idx_assistant_sessions_owner
    ON assistant_sessions (owner_type, owner_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_assistant_messages_session
    ON assistant_messages (session_id, sort_order);

-- Flattened view so the Streamlit appointment tracker is a single SELECT.
CREATE VIEW IF NOT EXISTS appointment_details AS
SELECT
    a.id            AS appointment_id,
    a.status        AS status,
    a.reason        AS reason,
    a.booked_at     AS booked_at,
    p.id            AS patient_id,
    p.full_name     AS patient_name,
    p.age           AS patient_age,
    d.id            AS doctor_id,
    d.full_name     AS doctor_name,
    d.specialty     AS specialty,
    d.hospital      AS hospital,
    s.slot_date     AS slot_date,
    s.slot_time     AS slot_time
FROM appointments a
JOIN patients p ON p.id = a.patient_id
JOIN doctors  d ON d.id = a.doctor_id
JOIN slots    s ON s.id = a.slot_id;
