"""Attendant view: family members and each person's medical history."""

from __future__ import annotations

import streamlit as st

from src.rag import build_patient_index, retrieve_patient_context
from src.tools.patient_tools import (
    ALLOWED_RELATIONS,
    add_medical_record,
    get_active_alerts,
    get_medical_history,
    get_patient_profile,
    list_patients,
    register_patient,
)
from src.ui.components import dataframe, metric_row, page_header

RECORD_TYPES = ("note", "diagnosis", "lab_result", "medication", "procedure")
ALERT_LEVELS = ("none", "low", "medium", "high")
GENDERS = ("", "Male", "Female", "Other")
SELECTED_KEY = "selected_patient_id"


def _register_member_form(attendant_id: int) -> None:
    with st.form("register_family_member", clear_on_submit=True):
        st.caption(
            "Register a relative so the assistant can keep their history and book "
            "for them by name or relation (for example “my father” or “Ramesh”)."
        )
        left, right = st.columns(2)
        full_name = left.text_input("Full name")
        relation = right.selectbox("Relation to you", ALLOWED_RELATIONS, index=1)
        age = left.number_input("Age", min_value=0, max_value=120, value=40, step=1)
        gender = right.selectbox("Gender", GENDERS)
        blood_group = left.text_input("Blood group", placeholder="B+")
        allergies = right.text_input("Allergies", placeholder="Penicillin, or none")
        if st.form_submit_button("Register family member", type="primary"):
            if not (full_name or "").strip():
                st.error("Enter the family member's full name.")
                return
            result = register_patient(
                attendant_id=attendant_id,
                full_name=full_name,
                relation=relation,
                age=int(age) if age else None,
                gender=gender or None,
                blood_group=blood_group or None,
                allergies=allergies or None,
            )
            if result["success"]:
                with st.spinner("Indexing the new chart..."):
                    build_patient_index(rebuild=True)
                st.session_state[SELECTED_KEY] = result["patient_id"]
                st.success(
                    f"{result['full_name']} is registered as your {result['relation']} "
                    f"(patient {result['patient_id']})."
                )
                st.rerun()
            else:
                st.error(result["error"])


def _pick_from_grid(family: list[dict]) -> dict | None:
    """The selected grid row is the only patient in focus."""
    ids = [person["id"] for person in family]
    stored = st.session_state.get(SELECTED_KEY)
    if stored not in ids:
        st.session_state[SELECTED_KEY] = ids[0]

    head = st.columns((2.6, 1.3, 0.8, 1.3))
    head[0].caption("Name")
    head[1].caption("Relation")
    head[2].caption("Age")
    head[3].caption("")

    for person in family:
        selected = person["id"] == st.session_state[SELECTED_KEY]
        row = st.columns((2.6, 1.3, 0.8, 1.3))
        if row[0].button(
            person["full_name"],
            key=f"pick_member_{person['id']}",
            type="primary" if selected else "secondary",
            width="stretch",
        ):
            st.session_state[SELECTED_KEY] = person["id"]
            st.rerun()
        row[1].write(person["relation"])
        row[2].write(str(person.get("age") or "-"))
        row[3].write("Selected" if selected else "")

    return get_patient_profile(st.session_state[SELECTED_KEY])


def _add_history_form(patient_id: int, name: str) -> None:
    with st.form("add_medical_history", clear_on_submit=True):
        st.caption(f"This note is stored on {name}'s chart only.")
        left, right = st.columns(2)
        record_type = left.selectbox("Record type", RECORD_TYPES)
        alert_level = right.selectbox("Alert level", ALERT_LEVELS)
        condition = left.text_input("Condition")
        diagnosis = right.text_input("Diagnosis / finding")
        treatment = left.text_input("Treatment or plan")
        medications = right.text_input("Medications")
        notes = st.text_area("Notes (free text)", height=100)
        if st.form_submit_button("Save medical history", type="primary"):
            if not any([condition, diagnosis, treatment, medications, notes]):
                st.error("Fill at least one field.")
                return
            result = add_medical_record(
                patient_id=patient_id,
                condition=condition or None,
                diagnosis=diagnosis or None,
                treatment=treatment or None,
                medications=medications or None,
                notes=notes or None,
                record_type=record_type,
                alert_level=alert_level,
            )
            if result["success"]:
                with st.spinner("Re-indexing patient charts..."):
                    build_patient_index(rebuild=True)
                st.success(f"Record {result['record_id']} saved on {name}'s chart.")
                st.rerun()
            else:
                st.error(result["error"])


def render() -> None:
    page_header(
        "Patient Records",
        "Select a family member in the grid, then view or add that person's history.",
    )
    attendant_id = st.session_state["attendant_id"]
    family = list_patients(attendant_id)

    st.subheader("Family members")
    if not family:
        st.info("No family members yet. Register the first person below.")
        with st.expander("Add a family member", expanded=True):
            _register_member_form(attendant_id)
        return

    profile = _pick_from_grid(family)
    with st.expander("Add a family member"):
        _register_member_form(attendant_id)

    if not profile:
        st.info("Select a family member in the grid.")
        return

    name = profile["full_name"]
    st.subheader(f"{name} · {profile['relation']}")
    st.caption("Medical history and new records below belong only to this member.")

    history = get_medical_history(profile["id"], limit=100)
    alerts = get_active_alerts(profile["id"])

    chart, add, search = st.tabs(
        ["Medical history", "Add medical history", "Search this chart"]
    )

    with chart:
        metric_row(
            [
                ("Age", profile.get("age") or "-"),
                ("Blood group", profile.get("blood_group") or "-"),
                ("Records", len(history)),
                ("Active alerts", len(alerts)),
            ]
        )
        if profile.get("allergies") and profile["allergies"].lower() != "none":
            st.error(f"**Allergies:** {profile['allergies']}", icon=":material/warning:")
        high = [item for item in alerts if item["alert_level"] == "high"]
        if high:
            st.warning(
                "**High-severity conditions:** "
                + ", ".join(sorted({item["condition"] for item in high})),
                icon=":material/priority_high:",
            )
        dataframe(
            [
                {
                    "Date": record["record_date"],
                    "Type": record["record_type"],
                    "Condition": record["condition"] or "-",
                    "Diagnosis": record["diagnosis"] or "-",
                    "Treatment": record["treatment"] or "-",
                    "Medications": record["medications"] or "-",
                    "Alert": record["alert_level"],
                }
                for record in history
            ],
            f"No medical history recorded yet for {name}.",
        )
        for record in history:
            if record["notes"]:
                with st.expander(
                    f"Notes - {record['record_date']} ({record['record_type']})"
                ):
                    st.write(record["notes"])

    with add:
        _add_history_form(profile["id"], name)

    with search:
        st.caption(f"Semantic search over {name}'s chart only.")
        query = st.text_input("Search", placeholder="kidney getting worse")
        if query:
            hits = retrieve_patient_context(profile["id"], query, k=4)
            if not hits:
                st.info("No matching records. The index may need rebuilding.")
            for hit in hits:
                st.markdown(
                    f"**{hit['record_date'] or 'profile'}** "
                    f"({hit['record_type']}) - distance `{hit['score']}`"
                )
                st.write(hit["text"])
                st.divider()
