"""Doctor Medical Information: search a patient and open their history."""

from __future__ import annotations

from collections import Counter

import streamlit as st

from src.tools.patient_tools import (
    get_active_alerts,
    get_medical_history,
    get_patient_profile,
    search_patients,
)
from src.ui.components import dataframe, metric_row, page_header

SELECTED_KEY = "doctor_search_patient_id"


def _render_chart(patient_id: int) -> None:
    profile = get_patient_profile(patient_id)
    if not profile:
        st.error("That patient is no longer on file.")
        return

    history = get_medical_history(patient_id, limit=100)
    alerts = get_active_alerts(patient_id)

    st.subheader(f"{profile['full_name']} · id {profile['id']}")
    if profile.get("allergies") and profile["allergies"].lower() != "none":
        st.error(f"**Allergy alert:** {profile['allergies']}", icon=":material/warning:")

    high = [item for item in alerts if item["alert_level"] == "high"]
    if high:
        st.warning(
            "**High-severity conditions:** "
            + ", ".join(sorted({item["condition"] for item in high if item["condition"]})),
            icon=":material/priority_high:",
        )

    conditions = Counter(record["condition"] for record in history if record["condition"])
    metric_row(
        [
            ("Age", profile.get("age") or "-"),
            ("Blood group", profile.get("blood_group") or "-"),
            ("Records", len(history)),
            ("Alerts", len(alerts)),
        ]
    )
    if conditions:
        st.caption(
            "Conditions on record: "
            + ", ".join(f"{name} ({count})" for name, count in conditions.most_common())
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
        f"No medical history recorded yet for {profile['full_name']}.",
    )
    for record in history:
        if record["notes"]:
            with st.expander(f"Notes — {record['record_date']} ({record['record_type']})"):
                st.write(record["notes"])


def render() -> None:
    page_header(
        "Medical Information",
        "Search a patient by name, id, or allergy, then open their history.",
    )

    query = st.text_input(
        "Search patients",
        placeholder="Ramesh, Sunita, sulfa, or a patient id",
    )
    matches = search_patients(query) if (query or "").strip() else []

    if (query or "").strip() and not matches:
        st.info("No patient matched that search.")
        return

    if not (query or "").strip():
        st.caption("Try a name such as Ramesh, an allergy such as penicillin, or a numeric id.")
        return

    ids = [person["id"] for person in matches]
    stored = st.session_state.get(SELECTED_KEY)
    if stored not in ids:
        st.session_state[SELECTED_KEY] = ids[0]

    st.caption(f"{len(matches)} match(es). Select a row to open the chart.")
    head = st.columns((2.8, 1.0, 2.2, 1.2))
    head[0].caption("Name")
    head[1].caption("Age")
    head[2].caption("Allergies")
    head[3].caption("")

    for person in matches:
        selected = person["id"] == st.session_state[SELECTED_KEY]
        row = st.columns((2.8, 1.0, 2.2, 1.2))
        if row[0].button(
            person["full_name"],
            key=f"doc_pick_{person['id']}",
            type="primary" if selected else "secondary",
            width="stretch",
        ):
            st.session_state[SELECTED_KEY] = person["id"]
            st.rerun()
        row[1].write(str(person.get("age") or "-"))
        row[2].write(person.get("allergies") or "none")
        row[3].write("Selected" if selected else "")

    _render_chart(st.session_state[SELECTED_KEY])
