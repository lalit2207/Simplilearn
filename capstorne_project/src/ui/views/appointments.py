"""Attendant view: appointment tracking and manual booking.

The capstone asks for "real-time appointment tracking". Every read here goes
straight to SQLite, so a booking made by the agent shows up on the next rerun
without any cache to invalidate.
"""

from __future__ import annotations

import streamlit as st

from src.tools.appointment_tools import (
    book_appointment,
    cancel_appointment,
    find_available_slots,
    list_appointments,
    list_specialties,
)
from src.tools.patient_tools import list_patients
from src.ui.components import dataframe, metric_row, page_header


def render() -> None:
    page_header("Appointments", "Track, book, and cancel appointments.")
    attendant_id = st.session_state["attendant_id"]

    patients = list_patients(attendant_id)
    if not patients:
        st.info(
            "No family members are registered yet. Add them on the Patient Records "
            "page, then you can book and track appointments here."
        )
        return

    patient_ids = {p["id"] for p in patients}
    appointments = [
        row for row in list_appointments() if row["patient_id"] in patient_ids
    ]
    confirmed = [a for a in appointments if a["status"] == "confirmed"]

    metric_row(
        [
            ("Total", len(appointments)),
            ("Confirmed", len(confirmed)),
            ("Cancelled", sum(1 for a in appointments if a["status"] == "cancelled")),
            ("Patients", len(patients)),
        ]
    )

    tracker, booking = st.tabs(["Tracker", "Book manually"])

    with tracker:
        dataframe(
            [
                {
                    "ID": a["appointment_id"],
                    "Patient": a["patient_name"],
                    "Doctor": a["doctor_name"],
                    "Specialty": a["specialty"],
                    "Hospital": a["hospital"],
                    "Date": a["slot_date"],
                    "Time": a["slot_time"],
                    "Status": a["status"],
                    "Reason": a["reason"] or "-",
                }
                for a in appointments
            ],
            "No appointments booked yet. Ask the assistant to book one.",
        )

        if confirmed:
            st.subheader("Cancel an appointment")
            labels = {
                f"#{a['appointment_id']} - {a['patient_name']} with {a['doctor_name']} "
                f"on {a['slot_date']} {a['slot_time']}": a["appointment_id"]
                for a in confirmed
            }
            choice = st.selectbox("Appointment", list(labels))
            if st.button("Cancel appointment", type="secondary"):
                result = cancel_appointment(labels[choice])
                if result["success"]:
                    st.success("Cancelled. The slot is available again.")
                    st.rerun()
                else:
                    st.error(result["error"])

    with booking:
        st.caption("Booking here uses the same tool the agent calls.")
        left, right = st.columns(2)
        patient_labels = {f"{p['full_name']} ({p['relation']})": p["id"] for p in patients}
        patient_choice = left.selectbox("Patient", list(patient_labels))
        specialty = right.selectbox("Specialty", list_specialties())

        slots = find_available_slots(specialty=specialty, limit=25)
        if not slots:
            st.info("No open slots for that specialty.")
            return

        slot_labels = {
            f"{s['slot_date']} {s['slot_time']} - {s['doctor_name']} "
            f"({s['hospital']}, rating {s['rating']})": s["slot_id"]
            for s in slots
        }
        slot_choice = st.selectbox("Available slot", list(slot_labels))
        reason = st.text_input("Reason for the visit", placeholder="CKD stage 3b review")

        if st.button("Book appointment", type="primary"):
            result = book_appointment(
                patient_id=patient_labels[patient_choice],
                slot_id=slot_labels[slot_choice],
                reason=reason or None,
            )
            if result["success"]:
                st.success(
                    f"Appointment {result['appointment_id']} confirmed with "
                    f"{result['doctor_name']} on {result['slot_date']} at {result['slot_time']}."
                )
                st.rerun()
            else:
                st.error(result["error"])
