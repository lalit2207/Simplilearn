"""Reusable render helpers.

The trace renderer here is the heart of the "how was this request processed"
requirement: it turns one AgentTrace into the plan, every tool call with its
arguments and output, and the final answer.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import llm_available, settings
from src.ui.branding import brand_markup

ALERT_COLOURS = {"high": "#c0392b", "medium": "#d68910", "low": "#2471a3", "none": "#566573"}

# Hide Streamlit chrome and Streamlit's own sidebar. Navigation is drawn by
# render_app_nav() so Chrome cannot hide the menu by collapsing the sidebar.
CHROME_CSS = """
<style>
#MainMenu,
[data-testid="stDecoration"],
[data-testid="stToolbar"],
[data-testid="stToolbarActions"],
.stDeployButton,
div[data-testid="stStatusWidget"],
[data-testid="stSidebar"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
    pointer-events: none !important;
}

.app-navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 56px;
    background: #ffffff;
    border-bottom: 1px solid #e5e7eb;
    z-index: 99990;
    display: flex;
    align-items: center;
    padding: 0 4.2rem 0 1.4rem;
}
.app-navbar .brand {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-weight: 700;
    font-size: 1.02rem;
    color: #111827 !important;
    text-decoration: none;
    letter-spacing: -0.01em;
    cursor: pointer;
}
.app-navbar img.app-logo,
.app-navbar .brand img.app-logo {
    width: 36px !important;
    height: 36px !important;
    max-width: 36px !important;
    max-height: 36px !important;
    border-radius: 8px;
    object-fit: contain;
    background: #fff;
}

[data-testid="stElementContainer"]:has(.brand-home-mark) {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stElementContainer"]:has(.brand-home-mark) + [data-testid="stElementContainer"],
[data-testid="stElementContainer"]:has(.brand-home-mark) + [data-testid="stLayoutWrapper"],
[data-testid="stLayoutWrapper"]:has(.brand-home-mark) + [data-testid="stElementContainer"],
[data-testid="stLayoutWrapper"]:has(.brand-home-mark) + [data-testid="stLayoutWrapper"] {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    z-index: 99996 !important;
    width: 320px !important;
    max-width: 42vw !important;
    height: 56px !important;
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stElementContainer"]:has(.brand-home-mark) + [data-testid="stElementContainer"] button,
[data-testid="stElementContainer"]:has(.brand-home-mark) + [data-testid="stLayoutWrapper"] button,
[data-testid="stLayoutWrapper"]:has(.brand-home-mark) + [data-testid="stElementContainer"] button,
[data-testid="stLayoutWrapper"]:has(.brand-home-mark) + [data-testid="stLayoutWrapper"] button {
    width: 100% !important;
    height: 56px !important;
    opacity: 0 !important;
    cursor: pointer !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.block-container { padding-top: 3.6rem !important; }

[data-testid="stHorizontalBlock"]:has(.app-sidenav-mark) {
    align-items: flex-start;
    gap: 0 !important;
}
[data-testid="stHorizontalBlock"]:has(.app-sidenav-mark) > div:first-child {
    background: #f7f8fa;
    border-right: 1px solid #e5e7eb;
    padding: 0.85rem 0.65rem 1.4rem 0.65rem;
    min-height: calc(100vh - 3.6rem);
    margin-left: -50px;
}
[data-testid="stHorizontalBlock"]:has(.app-sidenav-mark) > div:last-child {
    margin-left: 50px;
}
[data-testid="stHorizontalBlock"]:has(.app-sidenav-mark) > div:first-child a {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    padding: 0.55rem 0.75rem !important;
    border-radius: 8px !important;
    color: #374151 !important;
    text-decoration: none !important;
    font-weight: 500 !important;
}
[data-testid="stHorizontalBlock"]:has(.app-sidenav-mark) > div:first-child a:hover {
    background: #eef2f4 !important;
    color: #134e4e !important;
}
[data-testid="stHorizontalBlock"]:has(.app-sidenav-mark) > div:first-child a[aria-current="page"] {
    background: #e6f2f2 !important;
    color: #134e4e !important;
}
[data-testid="stHorizontalBlock"]:has(.assistant-nav-mark) {
    align-items: center !important;
    gap: 0 !important;
    margin: 0 !important;
}
[data-testid="stElementContainer"]:has(.assistant-nav-plus-mark) + [data-testid="stElementContainer"] button,
[data-testid="stElementContainer"]:has(.assistant-nav-plus-mark) + [data-testid="stLayoutWrapper"] button,
[data-testid="stLayoutWrapper"]:has(.assistant-nav-plus-mark) + [data-testid="stElementContainer"] button,
[data-testid="stLayoutWrapper"]:has(.assistant-nav-plus-mark) + [data-testid="stLayoutWrapper"] button {
    min-width: 32px !important;
    width: 32px !important;
    max-width: 32px !important;
    min-height: 32px !important;
    height: 32px !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    color: #374151 !important;
}
[data-testid="stElementContainer"]:has(.assistant-nav-plus-mark) + [data-testid="stElementContainer"] button:hover,
[data-testid="stElementContainer"]:has(.assistant-nav-plus-mark) + [data-testid="stLayoutWrapper"] button:hover,
[data-testid="stLayoutWrapper"]:has(.assistant-nav-plus-mark) + [data-testid="stElementContainer"] button:hover,
[data-testid="stLayoutWrapper"]:has(.assistant-nav-plus-mark) + [data-testid="stLayoutWrapper"] button:hover {
    background: #e6f2f2 !important;
    color: #134e4e !important;
}
[data-testid="stElementContainer"]:has(.assistant-history-mark),
[data-testid="stLayoutWrapper"]:has(.assistant-history-mark) {
    margin: 0.15rem 0 0.1rem 0 !important;
    min-height: 0 !important;
}
[data-testid="stHorizontalBlock"]:has(.app-sidenav-mark) > div:first-child .assistant-history button,
[data-testid="stHorizontalBlock"]:has(.app-sidenav-mark) > div:first-child [data-testid="stElementContainer"]:has(.assistant-history-mark) ~ [data-testid="stElementContainer"] button,
[data-testid="stHorizontalBlock"]:has(.app-sidenav-mark) > div:first-child [data-testid="stLayoutWrapper"]:has(.assistant-history-mark) ~ [data-testid="stLayoutWrapper"] button,
[data-testid="stHorizontalBlock"]:has(.app-sidenav-mark) > div:first-child [data-testid="stLayoutWrapper"]:has(.assistant-history-mark) ~ [data-testid="stElementContainer"] button {
    width: 100% !important;
    justify-content: flex-start !important;
    text-align: left !important;
    min-height: 28px !important;
    height: auto !important;
    padding: 0.2rem 0.55rem 0.2rem 1.15rem !important;
    margin: 0 !important;
    border: none !important;
    border-radius: 6px !important;
    background: transparent !important;
    box-shadow: none !important;
    color: #4b5563 !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
}
[data-testid="stHorizontalBlock"]:has(.app-sidenav-mark) > div:first-child [data-testid="stElementContainer"]:has(.assistant-history-mark) ~ [data-testid="stElementContainer"] button:hover,
[data-testid="stHorizontalBlock"]:has(.app-sidenav-mark) > div:first-child [data-testid="stLayoutWrapper"]:has(.assistant-history-mark) ~ [data-testid="stLayoutWrapper"] button:hover {
    background: #eef2f4 !important;
    color: #134e4e !important;
}
[data-testid="stHorizontalBlock"]:has(.app-sidenav-mark) > div:first-child [data-testid="stButton"] button p {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

div[data-testid="stPopover"] {
    position: fixed !important;
    top: 10px !important;
    right: 18px !important;
    left: auto !important;
    width: 36px !important;
    max-width: 36px !important;
    height: 36px !important;
    z-index: 100004;
    pointer-events: auto !important;
}
div[data-testid="stPopover"] > button,
[data-testid="stPopoverButton"] {
    width: 36px !important;
    max-width: 36px !important;
    min-width: 36px !important;
    height: 36px !important;
    min-height: 36px !important;
    border-radius: 50% !important;
    border: 1px solid #e5e7eb !important;
    background: #ffffff !important;
    color: #374151 !important;
}
[data-testid="stElementContainer"]:has(> div[data-testid="stPopover"]),
[data-testid="stElementContainer"]:has(> .stPopover) {
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: visible !important;
}
</style>
"""

HEADER_CSS = ""


def metric_row(items: list[tuple[str, object]], columns: int | None = None) -> None:
    """Render label/value pairs as a row of st.metric cards."""
    if not items:
        return
    for column, (label, value) in zip(st.columns(columns or len(items)), items):
        column.metric(label, value)


def alert_badge(level: str) -> str:
    colour = ALERT_COLOURS.get((level or "none").lower(), ALERT_COLOURS["none"])
    return f":{'red' if colour == ALERT_COLOURS['high'] else 'orange' if colour == ALERT_COLOURS['medium'] else 'blue'}[{level}]"


def dataframe(rows: list[dict], empty_message: str = "Nothing to show yet.", **kwargs) -> None:
    """Show rows as a table, or an info box when there are none."""
    if not rows:
        st.info(empty_message)
        return
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, **kwargs)


def render_plan(plan, expanded: bool = True) -> None:
    """Show the goal decomposition produced before execution."""
    if plan is None:
        st.caption("No plan was produced for this request.")
        return

    with st.expander(
        f"Planning breakdown - {len(plan.steps)} sub-goals (source: {plan.source})",
        expanded=expanded,
    ):
        st.markdown(f"**Goal:** {plan.goal}")
        for step in plan.steps:
            tool = "no tool (reasoning only)" if step.tool == "none" else f"`{step.tool}`"
            st.markdown(f"**{step.step}.** {step.sub_goal}  \n&nbsp;&nbsp;Tool: {tool}")
            if step.why:
                st.caption(f"Why: {step.why}")


def render_tool_calls(trace) -> None:
    """Show every tool invocation with arguments and raw output."""
    if not trace.tool_calls:
        st.caption("No tools were called for this request.")
        return

    st.markdown(f"**Tool calls ({len(trace.tool_calls)})**")
    for index, call in enumerate(trace.tool_calls, start=1):
        status = "ok" if call.ok else "FAILED"
        icon = ":material/check_circle:" if call.ok else ":material/error:"
        with st.expander(f"{index}. {call.name} - {status}", icon=icon):
            st.caption("Arguments")
            st.json(call.arguments or {}, expanded=False)
            st.caption("Output returned to the model")
            st.code(call.output or "(empty)", language="text")


def render_trace(trace, show_answer: bool = True, expanded: bool = False) -> None:
    """Full processing view for one request: plan, tools, timing, answer."""
    metric_row(
        [
            ("Tools called", len(trace.tool_calls)),
            ("Tool failures", len(trace.failed_calls)),
            ("Duration", f"{trace.duration_ms / 1000:.1f}s"),
            ("Outcome", "success" if trace.succeeded else "failed"),
        ]
    )

    render_plan(trace.plan, expanded=expanded)
    render_tool_calls(trace)

    if trace.error:
        st.error(trace.error)
    elif show_answer and trace.answer:
        st.markdown("**Final answer given to the attendant**")
        st.markdown(trace.answer)


def llm_warning() -> bool:
    """Warn when the Groq key is missing. Returns True when the LLM is usable."""
    if llm_available():
        return True
    st.warning(
        "**Groq API key not configured.** Set `GROQ_API_KEY` in `.env` (local) or "
        "in Streamlit secrets (hosted) and restart. "
        "Records, appointments, search and metrics still work without it.",
        icon=":material/key_off:",
    )
    return False


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


@st.dialog("Settings", width="large")
def settings_dialog() -> None:
    """Read-only model settings. The API key is never edited from the UI."""
    st.caption(
        "These values come from `.env` locally or Streamlit secrets when hosted. "
        "The API key is never shown."
    )

    left, right = st.columns(2)
    left.metric("Agent model", settings.groq_model.split("/")[-1])
    right.metric("Judge model", settings.groq_eval_model.split("/")[-1])
    st.text(
        f"Agent model      : {settings.groq_model}\n"
        f"Judge model      : {settings.groq_eval_model}\n"
        f"Temperature      : {settings.groq_temperature}\n"
        f"Max tokens       : {settings.groq_max_tokens}\n"
        f"Timeout          : {settings.groq_timeout}s\n"
        f"Embedding model  : {settings.embedding_model}\n"
        f"Vector store     : {settings.vector_db.upper()} (top k {settings.faiss_top_k})"
    )


def hide_streamlit_chrome() -> None:
    st.markdown(CHROME_CSS, unsafe_allow_html=True)


def render_app_nav(pages: list) -> None:
    """Always-visible left menu. Does not use Streamlit's collapsible sidebar."""
    from src.ui.state import (
        can_start_new_assistant,
        current_assistant_session_id,
        current_role,
        ensure_assistant_session,
        list_saved_assistant_sessions,
        open_assistant_session,
        start_new_doctor_thread,
        start_new_thread,
    )

    ensure_assistant_session()
    st.markdown('<div class="app-sidenav-mark"></div>', unsafe_allow_html=True)
    for page in pages:
        if getattr(page, "title", None) == "Assistant" and can_start_new_assistant():
            st.markdown('<div class="assistant-nav-mark"></div>', unsafe_allow_html=True)
            label_col, plus_col = st.columns([0.82, 0.18], vertical_alignment="center")
            with label_col:
                st.page_link(page, width="stretch")
            with plus_col:
                st.markdown('<div class="assistant-nav-plus-mark"></div>', unsafe_allow_html=True)
                if st.button(
                    "",
                    icon=":material/add:",
                    help="New assistant",
                    width="stretch",
                    key="nav_new_assistant",
                ):
                    if current_role() == "Doctor":
                        start_new_doctor_thread()
                    else:
                        start_new_thread()
                    st.switch_page(page)
            sessions = list_saved_assistant_sessions()
            if sessions:
                st.markdown('<div class="assistant-history-mark"></div>', unsafe_allow_html=True)
                active_id = current_assistant_session_id()
                for row in sessions:
                    title = row.get("title") or "New assistant"
                    selected = row["id"] == active_id
                    label = f"• {title}" if selected else title
                    if st.button(
                        label,
                        key=f"assistant_hist_{row['id']}",
                        type="tertiary",
                        width="stretch",
                        help=title,
                    ):
                        open_assistant_session(row["id"])
                        st.switch_page(page)
            continue
        st.page_link(page, width="stretch")


def app_banner() -> None:
    """Full-width website header: app name (home) on the left, user icon on the right."""
    from src.ui.state import (
        clear_role,
        current_role,
        current_user_name,
        is_guest,
        reset_attendant_access,
        reset_doctor_access,
    )

    if st.query_params.get("home") == "1":
        clear_role()
        st.query_params.clear()
        st.rerun()

    role = current_role()
    name = current_user_name() if role else "Guest"

    st.markdown(
        f'<nav class="app-navbar">{brand_markup()}</nav>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="brand-home-mark"></div>', unsafe_allow_html=True)
    if st.button("Home", key="brand_go_home", type="tertiary", help="Home"):
        clear_role()
        st.rerun()
    with st.popover("", icon=":material/account_circle:", help=name):
        if role:
            st.markdown(f"**{name}**")
            st.caption("Guest (not signed in)" if is_guest() else f"Signed in as {role}")
        else:
            st.caption("Choose a role on the home page to sign in.")
        if is_guest() and st.button(
            "Sign in as registered user",
            icon=":material/badge:",
            width="stretch",
            key="header_guest_signin",
        ):
            reset_attendant_access()
            st.rerun()
        if is_guest() and st.button(
            "Sign up",
            icon=":material/person_add:",
            width="stretch",
            key="header_guest_signup",
        ):
            reset_attendant_access(open_signup=True)
            st.rerun()
        if role == "Attendant" and not is_guest() and st.button(
            "Change how you proceed",
            width="stretch",
            key="header_change_access",
        ):
            reset_attendant_access()
            st.rerun()
        if role == "Doctor" and st.button(
            "Change doctor",
            width="stretch",
            key="header_change_doctor",
        ):
            reset_doctor_access()
            st.rerun()
        if st.button("Settings", icon=":material/settings:", width="stretch"):
            settings_dialog()
