"""LangChain tool definitions for the agent.

Design rule: the functions in the sibling modules return *data* (dicts and
lists) for the UI, while the wrappers here return *text* for the LLM. Keeping
the two apart means the Streamlit pages never have to parse prose, and the
model never has to parse Python repr output.

Each wrapper also has to be honest about failure - an empty result is
reported as a sentence the model can reason about, never as an exception.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.tools import appointment_tools as appts
from src.tools import patient_tools as patients
from src.tools.medical_search import format_search_results, search_medical_information


# --------------------------------------------------------------------------
# Argument schemas: these become the JSON schema the LLM sees, so the
# descriptions matter as much as the types.
# --------------------------------------------------------------------------
class ResolvePatientArgs(BaseModel):
    reference: str = Field(
        description="Patient id, relation ('my father', 'dad'), or name ('Ramesh')."
    )
    attendant_id: int | None = Field(
        default=None,
        description="Id of the caregiver speaking. Required to resolve relations.",
    )


class ListFamilyArgs(BaseModel):
    attendant_id: int | None = Field(
        default=None,
        description="Id of the caregiver speaking. Lists only that person's family.",
    )


class RegisterPatientArgs(BaseModel):
    attendant_id: int = Field(description="Id of the caregiver who is registering the member.")
    full_name: str = Field(description="Family member's full name.")
    relation: str = Field(
        default="other",
        description="Relation to the attendant: self, father, mother, spouse, child, "
        "brother, sister, grandfather, grandmother, uncle, aunt, or other.",
    )
    age: int | None = Field(default=None, description="Age in years.")
    gender: str | None = Field(default=None, description="Male, Female, or other.")
    blood_group: str | None = Field(default=None, description="Blood group if known.")
    allergies: str | None = Field(default=None, description="Known allergies, or none.")


class PatientHistoryArgs(BaseModel):
    patient_id: int = Field(description="Numeric patient id from resolve_patient.")
    limit: int = Field(default=20, description="Maximum number of records to return.")


class AddRecordArgs(BaseModel):
    patient_id: int = Field(description="Numeric patient id.")
    condition: str | None = Field(default=None, description="Condition name, e.g. 'Hypertension'.")
    diagnosis: str | None = Field(default=None, description="Clinical finding or diagnosis text.")
    treatment: str | None = Field(default=None, description="Treatment or plan advised.")
    medications: str | None = Field(default=None, description="Medications with dose and frequency.")
    notes: str | None = Field(default=None, description="Free-text observation from the attendant.")
    record_type: str = Field(
        default="note",
        description="One of: diagnosis, lab_result, medication, procedure, note.",
    )
    alert_level: str = Field(
        default="none", description="Severity: none, low, medium, or high."
    )


class FindDoctorsArgs(BaseModel):
    specialty: str = Field(
        description="Specialty or plain-language need, e.g. 'Nephrologist' or 'kidney'."
    )
    limit: int = Field(default=5, description="Maximum number of doctors to return.")


class FindSlotsArgs(BaseModel):
    specialty: str | None = Field(default=None, description="Specialty to filter by.")
    doctor_id: int | None = Field(default=None, description="Restrict to one doctor.")
    on_or_after: str | None = Field(
        default=None, description="Earliest acceptable date as YYYY-MM-DD."
    )
    limit: int = Field(default=8, description="Maximum number of slots to return.")


class BookArgs(BaseModel):
    patient_id: int = Field(description="Numeric patient id.")
    slot_id: int = Field(description="slot_id taken from find_available_slots.")
    reason: str | None = Field(default=None, description="Reason for the visit.")


class ListAppointmentsArgs(BaseModel):
    patient_id: int | None = Field(default=None, description="Filter to one patient.")
    status: str | None = Field(
        default=None, description="Filter by confirmed, cancelled, or completed."
    )


class CancelArgs(BaseModel):
    appointment_id: int = Field(description="Id of the appointment to cancel.")


class SearchArgs(BaseModel):
    query: str = Field(
        description="Disease, symptom, or treatment topic, e.g. 'chronic kidney disease treatment'."
    )
    max_results: int = Field(default=4, description="Maximum number of sources to return.")


class SearchPatientsArgs(BaseModel):
    query: str = Field(
        description="Patient name, numeric id, or allergy text, e.g. 'Ramesh' or 'sulfa'."
    )
    limit: int = Field(default=10, description="Maximum number of matching patients.")


class DoctorAppointmentsArgs(BaseModel):
    doctor_id: int = Field(description="Signed-in doctor's id. Always pass the current doctor_id.")
    patient_id: int | None = Field(default=None, description="Filter to one patient.")
    status: str | None = Field(
        default=None, description="Filter by confirmed, cancelled, or completed."
    )


# --------------------------------------------------------------------------
# Wrappers: data in, prose out.
# --------------------------------------------------------------------------
def _family_line(person: dict) -> str:
    return (
        f"{person['full_name']} (id {person['id']}, {person.get('relation')}, "
        f"age {person.get('age') or '-'})"
    )


def _resolve_patient(reference: str, attendant_id: int | None = None) -> str:
    roster = patients.format_family_roster(attendant_id)
    matches = patients.match_patients(reference, attendant_id)
    if len(matches) == 1:
        patient = matches[0]
        return (
            f"Patient identified: {patient['full_name']} (id {patient['id']}), "
            f"age {patient.get('age')}, {patient.get('gender')}, "
            f"relation to attendant: {patient.get('relation')}, "
            f"allergies: {patient.get('allergies') or 'none recorded'}."
        )
    if len(matches) > 1:
        names = "; ".join(_family_line(person) for person in matches)
        return (
            f"Several family members match '{reference}': {names}. "
            "Ask the attendant which person they mean before opening a chart or booking."
        )
    return (
        f"Could not identify a patient from '{reference}'. {roster} "
        "Ask which family member this is for, or offer to register a new member."
    )


def _list_family_members(attendant_id: int | None = None) -> str:
    members = patients.list_patients(attendant_id)
    if not members:
        return (
            "No family members are registered yet. Ask for a name, relation, age "
            "and allergies, then call register_patient — or send them to Patient Records."
        )
    lines = ["Registered family members. Ask which person if the request is about one of them:"]
    for person in members:
        lines.append(f"  - {_family_line(person)}")
    return "\n".join(lines)


def _register_patient(
    attendant_id: int,
    full_name: str,
    relation: str = "other",
    age: int | None = None,
    gender: str | None = None,
    blood_group: str | None = None,
    allergies: str | None = None,
) -> str:
    result = patients.register_patient(
        attendant_id=attendant_id,
        full_name=full_name,
        relation=relation,
        age=age,
        gender=gender,
        blood_group=blood_group,
        allergies=allergies,
    )
    if not result["success"]:
        return f"Could not register that family member: {result['error']}"
    return (
        f"Registered {result['full_name']} as your {result['relation']} "
        f"(patient id {result['patient_id']}). You can now ask about them by name or relation."
    )


def _get_patient_history(patient_id: int, limit: int = 20) -> str:
    digest = patients.build_history_digest(patient_id, limit=limit)
    alerts = patients.get_active_alerts(patient_id)
    if alerts:
        flagged = "; ".join(
            f"{a['condition']} ({a['alert_level']}, {a['record_date']})" for a in alerts
        )
        digest += f"\n\nACTIVE ALERTS: {flagged}"
    return digest


def _add_patient_record(**kwargs) -> str:
    result = patients.add_medical_record(**kwargs)
    if not result["success"]:
        return f"Failed to add record: {result['error']}"
    return (
        f"Record {result['record_id']} added to patient {result['patient_id']}'s chart."
    )


def _find_doctors(specialty: str, limit: int = 5) -> str:
    matches = appts.find_doctors(specialty, limit=limit)
    if not matches:
        available = ", ".join(appts.list_specialties())
        return (
            f"No doctors found for '{specialty}'. Available specialties: {available}."
        )
    lines = [f"Doctors matching '{specialty}':"]
    for doctor in matches:
        lines.append(
            f"  doctor_id={doctor['id']} | {doctor['full_name']} | {doctor['specialty']} "
            f"| {doctor['hospital']} | {doctor['experience_years']} yrs "
            f"| rating {doctor['rating']} | fee Rs.{doctor['consultation_fee']}"
        )
    return "\n".join(lines)


def _find_available_slots(
    specialty: str | None = None,
    doctor_id: int | None = None,
    on_or_after: str | None = None,
    limit: int = 8,
) -> str:
    slots = appts.find_available_slots(specialty, doctor_id, on_or_after, limit=limit)
    if not slots:
        return (
            "No available slots match that request. Try a different specialty, "
            "doctor, or a later start date."
        )
    lines = ["Available slots (use slot_id to book):"]
    for slot in slots:
        lines.append(
            f"  slot_id={slot['slot_id']} | {slot['slot_date']} {slot['slot_time']} "
            f"| {slot['doctor_name']} ({slot['specialty']}) | {slot['hospital']} "
            f"| rating {slot['rating']}"
        )
    return "\n".join(lines)


def _book_appointment(patient_id: int, slot_id: int, reason: str | None = None) -> str:
    result = appts.book_appointment(patient_id, slot_id, reason)
    if not result["success"]:
        return f"Booking failed: {result['error']}"
    return (
        f"Appointment {result['appointment_id']} confirmed for {result['patient_name']} "
        f"with {result['doctor_name']} ({result['specialty']}) at {result['hospital']} "
        f"on {result['slot_date']} at {result['slot_time']}."
    )


def _list_appointments(patient_id: int | None = None, status: str | None = None) -> str:
    rows = appts.list_appointments(patient_id=patient_id, status=status)
    if not rows:
        return "No appointments found for that filter."
    lines = ["Appointments:"]
    for row in rows:
        lines.append(
            f"  appointment_id={row['appointment_id']} | {row['patient_name']} "
            f"with {row['doctor_name']} ({row['specialty']}) "
            f"on {row['slot_date']} {row['slot_time']} | status {row['status']}"
        )
    return "\n".join(lines)


def _cancel_appointment(appointment_id: int) -> str:
    result = appts.cancel_appointment(appointment_id)
    if not result["success"]:
        return f"Cancellation failed: {result['error']}"
    return f"Appointment {appointment_id} cancelled and the slot released."


def _search_medical_information(query: str, max_results: int = 4) -> str:
    return format_search_results(search_medical_information(query, max_results))


def _search_patients(query: str, limit: int = 10) -> str:
    matches = patients.search_patients(query, limit=limit)
    if not matches:
        return (
            f"No patient matched '{query}'. Ask for a fuller name or a numeric id, "
            "or try an allergy such as penicillin."
        )
    if len(matches) == 1:
        person = matches[0]
        return (
            f"Patient identified: {person['full_name']} (id {person['id']}), "
            f"age {person.get('age')}, {person.get('gender')}, "
            f"allergies: {person.get('allergies') or 'none recorded'}."
        )
    lines = [f"Several patients match '{query}'. Ask which one, then use their id:"]
    for person in matches:
        lines.append(
            f"  - {person['full_name']} (id {person['id']}, age {person.get('age') or '-'}, "
            f"allergies: {person.get('allergies') or 'none'})"
        )
    return "\n".join(lines)


def _list_doctor_appointments(
    doctor_id: int,
    patient_id: int | None = None,
    status: str | None = None,
) -> str:
    rows = appts.list_appointments(
        patient_id=patient_id, doctor_id=doctor_id, status=status
    )
    if not rows:
        return "No appointments found on your list for that filter."
    lines = ["Your appointments:"]
    for row in rows:
        lines.append(
            f"  appointment_id={row['appointment_id']} | {row['patient_name']} "
            f"(patient_id={row['patient_id']}) on {row['slot_date']} {row['slot_time']} "
            f"| {row['reason'] or 'no reason'} | status {row['status']}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Tool objects
# --------------------------------------------------------------------------
RESOLVE_PATIENT_TOOL = StructuredTool.from_function(
    func=_resolve_patient,
    name="resolve_patient",
    description=(
        "Identify which family member the user means before any other patient action. "
        "Accepts an id, a relation such as 'my father', or a name. "
        "If several people match, the tool lists them so you can ask which one."
    ),
    args_schema=ResolvePatientArgs,
)

LIST_FAMILY_TOOL = StructuredTool.from_function(
    func=_list_family_members,
    name="list_family_members",
    description=(
        "List the attendant's registered family members. Use this when the request "
        "is about a person but they did not say a name or relation, then ask which "
        "member they mean."
    ),
    args_schema=ListFamilyArgs,
)

REGISTER_PATIENT_TOOL = StructuredTool.from_function(
    func=_register_patient,
    name="register_patient",
    description=(
        "Register a new family member so their medical history can be kept. "
        "Needs the attendant_id, full name, and relation."
    ),
    args_schema=RegisterPatientArgs,
)

GET_HISTORY_TOOL = StructuredTool.from_function(
    func=_get_patient_history,
    name="get_patient_history",
    description=(
        "Retrieve a patient's full medical history including diagnoses, treatments, "
        "medications, notes and active alerts. Requires a numeric patient_id."
    ),
    args_schema=PatientHistoryArgs,
)

ADD_RECORD_TOOL = StructuredTool.from_function(
    func=_add_patient_record,
    name="add_patient_record",
    description=(
        "Add a new record to a patient's chart. Use for both structured data "
        "(condition, diagnosis, medications) and free-text notes."
    ),
    args_schema=AddRecordArgs,
)

FIND_DOCTORS_TOOL = StructuredTool.from_function(
    func=_find_doctors,
    name="find_doctors",
    description=(
        "List doctors for a specialty. Accepts plain language such as 'kidney' "
        "or 'heart' as well as formal specialty names."
    ),
    args_schema=FindDoctorsArgs,
)

FIND_SLOTS_TOOL = StructuredTool.from_function(
    func=_find_available_slots,
    name="find_available_slots",
    description=(
        "Find open appointment slots by specialty, doctor, or earliest date. "
        "Always call this before book_appointment to obtain a valid slot_id."
    ),
    args_schema=FindSlotsArgs,
)

BOOK_TOOL = StructuredTool.from_function(
    func=_book_appointment,
    name="book_appointment",
    description=(
        "Book a specific slot for a patient. Needs a patient_id and a slot_id "
        "that came from find_available_slots."
    ),
    args_schema=BookArgs,
)

LIST_APPOINTMENTS_TOOL = StructuredTool.from_function(
    func=_list_appointments,
    name="list_appointments",
    description="List existing appointments, optionally filtered by patient or status.",
    args_schema=ListAppointmentsArgs,
)

CANCEL_TOOL = StructuredTool.from_function(
    func=_cancel_appointment,
    name="cancel_appointment",
    description="Cancel an appointment and return its slot to the calendar.",
    args_schema=CancelArgs,
)

SEARCH_TOOL = StructuredTool.from_function(
    func=_search_medical_information,
    name="search_medical_information",
    description=(
        "Look up current disease, symptom, or treatment information from "
        "MedlinePlus (US National Library of Medicine) and WHO fact sheets. "
        "Use for any general medical knowledge question. Results include URLs to cite."
    ),
    args_schema=SearchArgs,
)

SEARCH_PATIENTS_TOOL = StructuredTool.from_function(
    func=_search_patients,
    name="search_patients",
    description=(
        "Find a patient by name, id, or allergy so the doctor can open their chart. "
        "Call this before get_patient_history when the doctor names someone."
    ),
    args_schema=SearchPatientsArgs,
)

LIST_DOCTOR_APPOINTMENTS_TOOL = StructuredTool.from_function(
    func=_list_doctor_appointments,
    name="list_doctor_appointments",
    description=(
        "List the signed-in doctor's appointments. Always pass doctor_id. "
        "Optionally filter by patient_id or status."
    ),
    args_schema=DoctorAppointmentsArgs,
)

ALL_TOOLS = [
    RESOLVE_PATIENT_TOOL,
    LIST_FAMILY_TOOL,
    REGISTER_PATIENT_TOOL,
    GET_HISTORY_TOOL,
    ADD_RECORD_TOOL,
    FIND_DOCTORS_TOOL,
    FIND_SLOTS_TOOL,
    BOOK_TOOL,
    LIST_APPOINTMENTS_TOOL,
    CANCEL_TOOL,
    SEARCH_TOOL,
]

DOCTOR_TOOLS = [
    SEARCH_PATIENTS_TOOL,
    GET_HISTORY_TOOL,
    ADD_RECORD_TOOL,
    LIST_DOCTOR_APPOINTMENTS_TOOL,
    SEARCH_TOOL,
]

GUEST_TOOLS = [
    FIND_DOCTORS_TOOL,
    FIND_SLOTS_TOOL,
    SEARCH_TOOL,
]

TOOLS_BY_NAME = {tool.name: tool for tool in [*ALL_TOOLS, *DOCTOR_TOOLS]}


def tool_overview() -> list[dict]:
    """Name/description pairs - shown in the Streamlit tools panel."""
    return [{"name": tool.name, "description": tool.description} for tool in ALL_TOOLS]
