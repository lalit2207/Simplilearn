"""Doctor next step: pick which clinician to continue as.

The dropdown is a stand-in for a login id. It will be replaced later.
"""

from __future__ import annotations

import streamlit as st

from src.ui.branding import brand_markup
from src.ui.state import clear_role, continue_as_doctor, list_doctors
from src.ui.views.landing import HIDE_SIDEBAR_CSS


def render() -> None:
    st.markdown(HIDE_SIDEBAR_CSS, unsafe_allow_html=True)
    if st.button(
        "Back",
        icon=":material/arrow_back:",
        type="tertiary",
        key="back_to_landing_from_doctor",
    ):
        clear_role()
        st.rerun()
    st.markdown(
        f"""
        <div class="welcome-block">
          {brand_markup(size="hero")}
          <h1>Continue as a doctor</h1>
          <p class="vision">
            Choose your name from the clinic list. This dropdown stands in for a
            login id and will be replaced later.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, card, _ = st.columns([1, 1.4, 1], gap="medium")
    with card, st.container(border=True):
        st.markdown("### Clinician")
        st.caption("Select who you are")
        doctors = list_doctors()
        if not doctors:
            st.error("No doctors are registered in the clinic database.")
            return
        labels = {
            f"{row['full_name']}  ·  {row['specialty']}  ·  {row['hospital']}": row
            for row in doctors
        }
        choice = st.selectbox("Doctor", list(labels), key="doctor_user_pick")
        if st.button(
            "Continue to doctor module",
            icon=":material/stethoscope:",
            type="primary",
            width="stretch",
            key="proceed_doctor",
        ):
            person = labels[choice]
            continue_as_doctor(person["id"], person["full_name"])
            st.rerun()
