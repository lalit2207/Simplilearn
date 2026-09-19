"""Doctor calendar: add slots, remove slots, and block times."""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from src.tools.appointment_tools import (
    STANDARD_SLOT_TIMES,
    add_slot,
    list_doctor_slots,
    remove_slot,
    restore_slot,
    set_unavailable,
)
from src.ui.components import dataframe, metric_row, page_header


def _apply_times(
    doctor_id: int,
    slot_date: str,
    times: list[str],
    action: str,
) -> None:
    if not times:
        st.error("Choose at least one time.")
        return
    ok = 0
    errors: list[str] = []
    for slot_time in times:
        if action == "add":
            result = add_slot(doctor_id, slot_date, slot_time)
        else:
            result = set_unavailable(doctor_id, slot_date, slot_time)
        if result["success"]:
            ok += 1
        else:
            errors.append(f"{slot_time}: {result['error']}")
    if ok:
        verb = "opened" if action == "add" else "blocked"
        st.success(f"{ok} time(s) {verb} on {slot_date}.")
    for message in errors:
        st.warning(message)
    if ok:
        st.rerun()


def render() -> None:
    doctor_id = st.session_state.get("doctor_id", 1)
    page_header(
        "Availability",
        "Open new times, remove unused slots, or block a time so no one can book it.",
    )

    today = date.today().isoformat()
    all_slots = list_doctor_slots(doctor_id, on_or_after=today)
    available = [s for s in all_slots if s["status"] == "available"]
    booked = [s for s in all_slots if s["status"] == "booked"]
    blocked = [s for s in all_slots if s["status"] == "unavailable"]

    metric_row(
        [
            ("Open", len(available)),
            ("Booked", len(booked)),
            ("Blocked", len(blocked)),
        ]
    )

    add_tab, block_tab = st.tabs(["Add slots", "Block time"])
    default_day = date.today() + timedelta(days=1)
    while default_day.weekday() >= 5:
        default_day += timedelta(days=1)

    with add_tab:
        with st.form("add_slots_form"):
            st.caption("These times become bookable for attendants.")
            add_date = st.date_input("Date", value=default_day, min_value=date.today())
            add_times = st.multiselect("Times", STANDARD_SLOT_TIMES, default=["10:00", "14:00"])
            if st.form_submit_button("Add slots", type="primary"):
                _apply_times(doctor_id, add_date.isoformat(), add_times, "add")

    with block_tab:
        with st.form("block_slots_form"):
            st.caption(
                "Blocked times cannot be booked. If a slot is already booked, "
                "cancel that visit first."
            )
            block_date = st.date_input("Date", value=default_day, min_value=date.today(), key="block_date")
            block_times = st.multiselect(
                "Times to block", STANDARD_SLOT_TIMES, default=["16:00"], key="block_times"
            )
            if st.form_submit_button("Add unavailability", type="primary"):
                _apply_times(doctor_id, block_date.isoformat(), block_times, "block")

    st.subheader("Clinic times")
    day = st.date_input("Show date", value=default_day, min_value=date.today(), key="avail_day")
    day_slots = list_doctor_slots(doctor_id, slot_date=day.isoformat())

    if not day_slots:
        st.info(f"No slots on {day.isoformat()}. Add or block times above.")
    else:
        header = st.columns((1.2, 1.4, 2.4, 1.4, 1.4))
        header[0].caption("Time")
        header[1].caption("Status")
        header[2].caption("")
        header[3].caption("")
        header[4].caption("")

        for slot in day_slots:
            row = st.columns((1.2, 1.4, 2.4, 1.4, 1.4))
            row[0].write(slot["slot_time"])
            row[1].write(slot["status"])
            slot_id = slot["slot_id"]
            if slot["status"] == "available":
                if row[2].button("Remove", key=f"rm_{slot_id}"):
                    result = remove_slot(slot_id, doctor_id)
                    if result["success"]:
                        st.rerun()
                    else:
                        st.error(result["error"])
                if row[3].button("Block", key=f"blk_{slot_id}"):
                    result = set_unavailable(doctor_id, slot["slot_date"], slot["slot_time"])
                    if result["success"]:
                        st.rerun()
                    else:
                        st.error(result["error"])
            elif slot["status"] == "unavailable":
                if row[2].button("Restore", key=f"rst_{slot_id}"):
                    result = restore_slot(slot_id, doctor_id)
                    if result["success"]:
                        st.rerun()
                    else:
                        st.error(result["error"])
                if row[3].button("Remove", key=f"rmu_{slot_id}"):
                    result = remove_slot(slot_id, doctor_id)
                    if result["success"]:
                        st.rerun()
                    else:
                        st.error(result["error"])
            else:
                row[2].caption("Booked — cancel the visit to free this time.")

    st.caption("Upcoming days at a glance")
    by_date: dict[str, dict[str, int]] = {}
    for slot in all_slots:
        counts = by_date.setdefault(slot["slot_date"], {"available": 0, "booked": 0, "unavailable": 0})
        counts[slot["status"]] = counts.get(slot["status"], 0) + 1
    dataframe(
        [
            {
                "Date": day_key,
                "Open": counts.get("available", 0),
                "Booked": counts.get("booked", 0),
                "Blocked": counts.get("unavailable", 0),
            }
            for day_key, counts in sorted(by_date.items())
        ],
        "No upcoming slots.",
    )
