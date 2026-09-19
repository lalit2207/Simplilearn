"""Doctor dashboard: confirmed visits on the signed-in doctor's list."""

from __future__ import annotations

import streamlit as st

from src.tools.appointment_tools import list_appointments, list_doctor_slots
from src.ui.components import dataframe, metric_row, page_header
from src.ui.state import list_doctors


def render() -> None:
    doctor_id = st.session_state.get("doctor_id", 1)
    profile = next((row for row in list_doctors() if row["id"] == doctor_id), None)

    page_header(
        "My Dashboard",
        f"{profile['specialty']} at {profile['hospital']}"
        if profile
        else "Your schedule and the patients booked with you.",
    )

    appointments = list_appointments(doctor_id=doctor_id)
    confirmed = [row for row in appointments if row["status"] == "confirmed"]
    open_slots = list_doctor_slots(doctor_id, status="available")

    metric_row(
        [
            ("Confirmed appointments", len(confirmed)),
            ("Cancelled", sum(1 for row in appointments if row["status"] == "cancelled")),
            ("Open slots", len(open_slots)),
            ("Distinct patients", len({row["patient_id"] for row in confirmed})),
        ]
    )

    dataframe(
        [
            {
                "ID": row["appointment_id"],
                "Date": row["slot_date"],
                "Time": row["slot_time"],
                "Patient": row["patient_name"],
                "Age": row["patient_age"],
                "Reason": row["reason"] or "-",
                "Status": row["status"],
            }
            for row in appointments
        ],
        "No appointments booked with you yet.",
    )
    st.caption(
        "Ask the assistant about a patient, manage open times on Availability, "
        "or search a chart under Medical Information."
    )
