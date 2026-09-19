"""Agentic Healthcare Assistant - Streamlit entry point.

    streamlit run app.py

Navigation is role-based. The page set is built from the selected role, so an
attendant never sees agent internals and the admin sees everything:

    Attendant     - assistant chat, records, appointments, medical information
    Doctor        - own schedule and the charts of patients booked with them
    Admin/LLMOps  - traces, memory, evaluation metrics, system configuration
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `from src...` when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import settings  # noqa: E402
from src.ui.branding import LOGO_PATH  # noqa: E402
from src.ui.components import app_banner, hide_streamlit_chrome, render_app_nav  # noqa: E402
from src.ui.state import (  # noqa: E402
    attendant_ready,
    current_role,
    doctor_ready,
    ensure_database,
    init_state,
    is_guest,
)
from src.ui.views import (  # noqa: E402
    appointments,
    assistant,
    availability,
    doctor,
    doctor_assistant,
    doctor_search,
    evaluation,
    knowledge,
    landing,
    patients,
    system,
    traces,
)
from src.ui.views import attendant_entry, doctor_entry  # noqa: E402

st.set_page_config(
    page_title=settings.app_title,
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else ":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _pages_for(role: str) -> list:
    """Build the page set for a role.

    Every view's entry point is named `render`, and Streamlit infers a page's
    URL from the callable name - which would make all pathnames collide. So
    `url_path` is set explicitly on each page.
    """
    if role == "Doctor":
        return [
            st.Page(
                doctor_assistant.render,
                title="Assistant",
                icon=":material/forum:",
                url_path="doctor-assistant",
                default=True,
            ),
            st.Page(
                doctor.render,
                title="My Dashboard",
                icon=":material/stethoscope:",
                url_path="doctor",
            ),
            st.Page(
                availability.render,
                title="Availability",
                icon=":material/event_available:",
                url_path="availability",
            ),
            st.Page(
                doctor_search.render,
                title="Medical Information",
                icon=":material/menu_book:",
                url_path="doctor-knowledge",
            ),
        ]

    if role == "Admin / LLMOps":
        return [
            st.Page(
                evaluation.render,
                title="Evaluation & Metrics",
                icon=":material/analytics:",
                url_path="evaluation",
                default=True,
            ),
            st.Page(
                traces.render,
                title="Traces, Memory & Logs",
                icon=":material/account_tree:",
                url_path="traces",
            ),
            st.Page(
                system.render,
                title="System",
                icon=":material/settings:",
                url_path="system",
            ),
        ]

    pages = [
        st.Page(
            assistant.render,
            title="Assistant",
            icon=":material/forum:",
            url_path="assistant",
            default=True,
        ),
        st.Page(
            knowledge.render,
            title="Medical Information",
            icon=":material/menu_book:",
            url_path="knowledge",
        ),
    ]
    if is_guest():
        return pages
    return [
        pages[0],
        st.Page(
            patients.render,
            title="Patient Records",
            icon=":material/folder_shared:",
            url_path="patients",
        ),
        st.Page(
            appointments.render,
            title="Appointments",
            icon=":material/event:",
            url_path="appointments",
        ),
        pages[1],
    ]


def main() -> None:
    settings.ensure_directories()
    ensure_database()
    init_state()
    hide_streamlit_chrome()

    role = current_role()
    if role is None:
        landing.render()
        return
    if role == "Attendant" and not attendant_ready():
        attendant_entry.render()
        return
    if role == "Doctor" and not doctor_ready():
        doctor_entry.render()
        return

    app_banner()
    pages = _pages_for(role)
    current = st.navigation(pages, position="hidden")
    nav_col, main_col = st.columns([0.22, 0.78], gap="medium")
    with nav_col:
        render_app_nav(pages)
    with main_col:
        current.run()


main()
