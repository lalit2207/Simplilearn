"""Landing screen: choose a role before entering the app.

Picking Attendant or Doctor opens a proceed page to choose an identity.
Admin continues as a generic admin user. The header is drawn by `app.py`.
"""

from __future__ import annotations

import streamlit as st

from src.ui.branding import asset_data_uri, brand_markup
from src.ui.state import ROLES, set_role

HIDE_SIDEBAR_CSS = """
<style>
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"],
[data-testid="stToolbar"],
[data-testid="collapsedControl"],
.app-navbar,
header[data-testid="stHeader"],
div[data-testid="stPopover"] { display: none !important; }
.block-container { padding-top: 2.4rem !important; }
.welcome-block { text-align: center; padding: 0.4rem 14% 1rem 14%; }
.welcome-block .brand-mark {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.55rem;
    margin: 0 0 0.85rem 0;
}
.welcome-block .brand-mark img.app-logo {
    width: 112px;
    height: 112px;
    border-radius: 22px;
    object-fit: contain;
    background: #fff;
    box-shadow: 0 4px 16px rgba(26, 127, 127, 0.12);
}
.welcome-block .brand-mark span,
.welcome-block .app-name {
    margin: 0;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #1a7f7f;
}
.welcome-block h1 { font-size: 2.45rem; margin: 0 0 0.55rem 0; line-height: 1.15; }
.welcome-block .vision { color: #4b5563; font-size: 1.02rem; line-height: 1.55; margin: 0; }
</style>
"""

LANDING_CSS = """
<style>
.stApp {
    background:
        radial-gradient(ellipse 80% 40% at 50% -8%, rgba(26, 127, 127, 0.14), transparent 62%),
        linear-gradient(180deg, #f3faf9 0%, #ffffff 36%, #f7f9fb 100%) !important;
}
.block-container {
    padding-top: 0.85rem !important;
    padding-bottom: 2.6rem !important;
    max-width: 1160px !important;
}

.landing-top { text-align: center; padding: 0 4% 0.15rem 4%; }
.landing-top .brand-mark {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.45rem;
    margin: 0 0 0.45rem 0;
}
.landing-top .brand-mark img.app-logo {
    width: 64px;
    height: 64px;
    border-radius: 16px;
    object-fit: contain;
    background: #fff;
    box-shadow: 0 6px 18px rgba(26, 127, 127, 0.14);
}
.landing-top .brand-mark span {
    margin: 0;
    font-size: 1.02rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    text-transform: none;
    color: #1a7f7f;
}
.landing-top .landing-kicker { margin: 0 0 0.7rem 0; }
.landing-top .vision {
    color: #4b5563;
    font-size: 1.02rem;
    line-height: 1.55;
    margin: 0 auto 0.85rem auto;
    max-width: 46rem;
}

.landing-hero { text-align: center; padding: 1.6rem 8% 0.2rem 8%; }
.landing-hero .brand-mark {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    margin: 0 0 0.75rem 0;
}
.landing-hero .brand-mark img.app-logo {
    width: 72px;
    height: 72px;
    border-radius: 18px;
    object-fit: contain;
    background: #fff;
    box-shadow: 0 8px 24px rgba(26, 127, 127, 0.16);
}
.landing-hero .brand-mark span {
    margin: 0;
    font-size: 0.92rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #1a7f7f;
}
.landing-kicker {
    display: inline-block;
    margin: 0 0 0.7rem 0;
    padding: 0.22rem 0.7rem;
    border-radius: 999px;
    background: #e6f3f3;
    color: #134e4e;
    font-size: 0.78rem;
    font-weight: 650;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.landing-hero h1 {
    font-size: 2.55rem;
    margin: 0 0 0.7rem 0;
    line-height: 1.15;
    letter-spacing: -0.03em;
    color: #111827;
}
.landing-hero .vision {
    color: #4b5563;
    font-size: 1.05rem;
    line-height: 1.6;
    margin: 0 auto 1.15rem auto;
    max-width: 46rem;
}
.landing-pills {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0 0 1.35rem 0;
}
.landing-pills span {
    background: #fff;
    border: 1px solid #d7e4e4;
    color: #134e4e;
    border-radius: 999px;
    padding: 0.32rem 0.85rem;
    font-size: 0.82rem;
    font-weight: 600;
}
.landing-hero-art {
    width: min(920px, 100%);
    height: auto;
    border-radius: 22px;
    box-shadow: 0 18px 50px rgba(15, 23, 42, 0.10);
    border: 1px solid #e5eeed;
    margin: 0 auto 1.7rem auto;
    display: block;
    background: #fff;
}

.landing-features {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.9rem;
    margin: 0 0 0.4rem 0;
}
.feature-card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 1.05rem 1.1rem 1.15rem 1.1rem;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.04);
}
.feature-card .num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: #e6f3f3;
    color: #1a7f7f;
    font-size: 0.78rem;
    font-weight: 750;
    margin-bottom: 0.55rem;
}
.feature-card h3 {
    margin: 0 0 0.35rem 0;
    font-size: 1.02rem;
    color: #111827;
}
.feature-card p {
    margin: 0;
    color: #4b5563;
    font-size: 0.92rem;
    line-height: 1.5;
}

.landing-section {
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    margin: 0 0 0.55rem 0;
}
.landing-section .label {
    display: inline-block;
    width: auto;
    margin: 0 auto 0.45rem auto;
    text-align: center;
}
.landing-section h2 {
    margin: 0 auto 0.35rem auto;
    width: 100%;
    text-align: center;
    font-size: 1.45rem;
    letter-spacing: -0.02em;
    color: #111827;
}
.landing-section a[href^="#"] { display: none !important; }
.landing-section p {
    margin: 0 auto;
    max-width: 36rem;
    color: #6b7280;
    font-size: 0.95rem;
}

.role-card {
    margin-bottom: 0.15rem;
    min-height: 470px;
}
.role-card img.role-photo {
    width: 100%;
    height: 120px;
    object-fit: cover;
    border-radius: 14px 14px 0 0;
    display: block;
    background: #eef6f6;
}
.role-card .role-body { padding: 0.85rem 0.15rem 0.15rem 0.15rem; }
.role-card .kicker {
    margin: 0 0 0.25rem 0;
    color: #1a7f7f;
    font-size: 0.75rem;
    font-weight: 750;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.role-card h3 {
    margin: 0 0 0.35rem 0;
    font-size: 1.28rem;
    color: #111827;
}
.role-card .audience {
    margin: 0 0 0.55rem 0;
    color: #6b7280;
    font-size: 0.86rem;
}
.role-card .detail {
    margin: 0 0 0.7rem 0;
    min-height: 7.2em;
    color: #374151;
    font-size: 0.92rem;
    line-height: 1.5;
}
.role-card ul {
    margin: 0;
    padding: 0 0 0 1.05rem;
    min-height: 6.4em;
    color: #4b5563;
    font-size: 0.88rem;
    line-height: 1.55;
}

[data-testid="stHorizontalBlock"]:has(.role-card) {
    align-items: stretch !important;
    gap: 1.1rem !important;
}
[data-testid="stHorizontalBlock"]:has(.role-card) > div {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
    padding: 0.7rem 0.75rem 0.85rem 0.75rem !important;
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="stHorizontalBlock"]:has(.role-card) > div > div,
[data-testid="stHorizontalBlock"]:has(.role-card) [data-testid="stVerticalBlock"] {
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
}
[data-testid="stHorizontalBlock"]:has(.role-card) > div [data-testid="stVerticalBlockBorderWrapper"] {
    border: none !important;
}
[data-testid="stHorizontalBlock"]:has(.role-card) [data-testid="stButton"] {
    margin-top: auto !important;
    width: 100% !important;
}

.landing-foot {
    margin: 1.8rem 10% 0 10%;
    padding-top: 1rem;
    border-top: 1px solid #e5e7eb;
    text-align: center;
    color: #6b7280;
    font-size: 0.84rem;
    line-height: 1.5;
}
@media (max-width: 900px) {
    .landing-hero { padding-left: 2%; padding-right: 2%; }
    .landing-hero h1 { font-size: 2rem; }
    .landing-features { grid-template-columns: 1fr; }
}
</style>
"""

ROLE_CARDS = {
    "Attendant": {
        "icon": ":material/family_restroom:",
        "kicker": "Families",
        "user": "Guest or a registered caregiver",
        "image": "role-attendant.png",
        "detail": (
            "Look up specialists and medical information as a guest, or sign in "
            "to book visits and keep family charts."
        ),
        "pages": ["How you proceed", "Assistant", "Patient Records", "Appointments"],
    },
    "Doctor": {
        "icon": ":material/stethoscope:",
        "kicker": "Clinic",
        "user": "Choose a clinician from the list",
        "image": "role-doctor.png",
        "detail": (
            "Pick which doctor you are, then ask about a patient, manage open "
            "or blocked slots, and search any chart."
        ),
        "pages": ["Who you are", "Assistant", "My Dashboard", "Availability"],
    },
    "Admin / LLMOps": {
        "icon": ":material/monitoring:",
        "kicker": "Operations",
        "user": "Admin user",
        "image": "role-admin.png",
        "detail": (
            "Inspect how each request was planned, review agent memory, and run "
            "QAEvalChain evaluations with per-module success rates."
        ),
        "pages": ["Evaluation & Metrics", "Traces, Memory & Logs", "System"],
    },
}


def _role_markup(role: str, card: dict) -> str:
    pages = "".join(f"<li>{page}</li>" for page in card["pages"])
    return f"""
    <div class="role-card">
      <img class="role-photo" src="{asset_data_uri(card['image'])}" alt="{role}" />
      <div class="role-body">
        <p class="kicker">{card['kicker']}</p>
        <h3>{role}</h3>
        <p class="audience">{card['user']}</p>
        <p class="detail">{card['detail']}</p>
        <ul>{pages}</ul>
      </div>
    </div>
    """


def render() -> None:
    st.markdown(HIDE_SIDEBAR_CSS + LANDING_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="landing-top">
          {brand_markup(size="hero")}
          <p class="landing-kicker">Agentic care workspace</p>
          <p class="vision">
            A single assistant for booking visits, keeping histories current, and
            answering medical questions from reliable sources — so families,
            doctors, and operations work from the same record.
          </p>
        </div>
        <div class="landing-section">
          <p class="label landing-kicker">Get started</p>
          <h2>Continue in the role that fits your work</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for column, role in zip(st.columns(len(ROLES), gap="medium"), ROLES):
        card = ROLE_CARDS[role]
        with column:
            st.markdown(_role_markup(role, card), unsafe_allow_html=True)
            if st.button(
                f"Continue as {role.split(' /')[0]}",
                icon=card["icon"],
                type="primary",
                width="stretch",
                key=f"role_{role}",
            ):
                set_role(role)
                st.rerun()

    st.markdown(
        f"""
        <div class="landing-hero">
          <h1>One trusted picture of the patient</h1>
          <div class="landing-pills">
            <span>Book visits</span>
            <span>Family charts</span>
            <span>Cited medical answers</span>
            <span>Traceable agent runs</span>
          </div>
          <img class="landing-hero-art" src="{asset_data_uri('hero-clinic.png')}" alt="" />
        </div>
        <div class="landing-features">
          <div class="feature-card">
            <div class="num">01</div>
            <h3>Coordinate the visit</h3>
            <p>Find a specialist, see open times, and book without chasing phone trees or separate portals.</p>
          </div>
          <div class="feature-card">
            <div class="num">02</div>
            <h3>Keep the chart current</h3>
            <p>Family members and clinicians share one history, including allergies, notes, and past visits.</p>
          </div>
          <div class="feature-card">
            <div class="num">03</div>
            <h3>Show how the answer was made</h3>
            <p>Every run can be planned, tooled, and evaluated — so operations can see what the agent did.</p>
          </div>
        </div>
        <p class="landing-foot">
          Medical answers are drawn from MedlinePlus and the clinic record.
          This assistant supports care work — it does not replace a clinician.
        </p>
        """,
        unsafe_allow_html=True,
    )
