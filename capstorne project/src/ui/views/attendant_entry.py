"""Attendant next step: guest, registered caregiver, or sign up."""

from __future__ import annotations

import streamlit as st

from src.tools.patient_tools import register_attendant
from src.ui.branding import brand_markup
from src.ui.state import (
    clear_role,
    continue_as_guest,
    continue_as_registered,
    list_attendants,
)
from src.ui.views.landing import HIDE_SIDEBAR_CSS


def _signup_form(form_key: str = "signup_attendant") -> None:
    with st.form(form_key, clear_on_submit=False):
        full_name = st.text_input("Full name")
        email = st.text_input("Email")
        phone = st.text_input("Phone", placeholder="Optional")
        if st.form_submit_button("Create account", type="primary"):
            result = register_attendant(full_name, email=email, phone=phone)
            if not result["success"]:
                st.error(result["error"])
                return
            list_attendants.clear()
            st.session_state["open_signup"] = False
            continue_as_registered(result["attendant_id"], result["full_name"])
            st.rerun()


def render() -> None:
    st.markdown(HIDE_SIDEBAR_CSS, unsafe_allow_html=True)
    if st.button(
        "Back",
        icon=":material/arrow_back:",
        type="tertiary",
        key="back_to_landing",
    ):
        clear_role()
        st.rerun()
    st.markdown(
        f"""
        <div class="welcome-block">
          {brand_markup(size="hero")}
          <h1>How do you want to proceed?</h1>
          <p class="vision">
            Browse as a guest, sign in as a registered caregiver, or create a new
            account to book visits and keep family charts.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    guest_col, registered_col, signup_col = st.columns(3, gap="medium")

    with guest_col, st.container(border=True):
        st.markdown("### Guest")
        st.caption("No sign-in")
        st.write(
            "Look up specialists, open appointment times, and general information "
            "about medicines or diseases."
        )
        st.markdown(
            "**You will be asked to sign in** to book an appointment or open a patient history."
        )
        if st.button(
            "Continue as Guest",
            icon=":material/person:",
            type="primary",
            width="stretch",
            key="proceed_guest",
        ):
            st.session_state["open_signup"] = False
            continue_as_guest()
            st.rerun()

    with registered_col, st.container(border=True):
        st.markdown("### Registered user")
        st.caption("Already have an account")
        st.write("Open the family assistant for booking, charts, and records.")
        attendants = list_attendants()
        if not attendants:
            st.info("No accounts yet. Use Sign up.")
        else:
            labels = {
                f"{row['full_name']}  ·  {row.get('email') or 'no email'}": row
                for row in attendants
            }
            choice = st.selectbox("Registered user", list(labels), key="registered_user_pick")
            if st.button(
                "Continue as registered user",
                icon=":material/badge:",
                type="primary",
                width="stretch",
                key="proceed_registered",
            ):
                st.session_state["open_signup"] = False
                person = labels[choice]
                continue_as_registered(person["id"], person["full_name"])
                st.rerun()

    with signup_col, st.container(border=True):
        st.markdown("### Sign up")
        st.caption("New caregiver")
        st.write(
            "Create an account so you can book appointments and keep medical "
            "history for yourself and your family."
        )
        _signup_form()
