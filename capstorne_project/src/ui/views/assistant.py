"""Attendant view: talk to the agent.

The composer matches a ChatGPT-style bar: paperclip on the left, the typed
message in the middle, send on the right. New sessions start from the +
next to Assistant in the left menu (registered users only).
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from src.config import settings
from src.llmops import log_run
from src.ui.components import llm_warning, page_header, render_trace
from src.ui.composer_files import (
    ATTENDANT_FILES,
    ATTENDANT_PENDING_IMPORT,
    build_turn,
    render_attach_button,
    render_attach_picker,
    render_file_chips,
)
from src.ui.state import get_agent, get_memory, is_guest, persist_assistant_chat, record_trace

QUICK_REPLIES = {
    "Book a nephrologist + treatments": (
        "My 70-year-old father has chronic kidney disease. I want to book a "
        "nephrologist for him. Also, can you summarize latest treatment methods?"
    ),
    "Summarise history": "Summarise my father's medical history.",
    "Check allergies": (
        "Does my father have any allergies I should flag before his consultation?"
    ),
    "Find a specialist": "Which kidney specialists are available to see my father?",
    "Latest CKD treatments": (
        "What are the current treatment approaches for chronic kidney disease?"
    ),
    "Book for my mother": "Book a cardiologist for my mother sometime next week.",
    "Add a note": "Record that my father reported ankle swelling again this week.",
    "My appointments": "What appointments does my father have booked?",
}

GUEST_REPLIES = {
    "Find a nephrologist": "Which nephrologists are available, and where do they practise?",
    "Open slots": "What appointment times are open with a nephrologist this week?",
    "Latest CKD treatments": (
        "What are the current treatment approaches for chronic kidney disease?"
    ),
    "What is hypertension": "What is hypertension, in plain language?",
}

COMPOSER_KEY = "assistant_draft"

COMPOSER_CSS = """
<style>
.composer-anchor + div {
    border: 1px solid #d0d5dd;
    border-radius: 999px;
    padding: 0.15rem 0.4rem;
    background: #fff;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}
.composer-anchor + div [data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}
.composer-anchor + div [data-testid="stTextInput"] input {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}
.composer-anchor + div [data-testid="stTextInput"] > div {
    border: none !important;
    background: transparent !important;
}
.composer-anchor + div > div:first-child {
    flex: 0 0 42px !important;
    width: 42px !important;
    min-width: 42px !important;
    max-width: 42px !important;
}

[data-testid="stElementContainer"]:has(.chip-anchor) ~ [data-testid="stLayoutWrapper"] [data-testid="stHorizontalBlock"] {
    flex-wrap: wrap !important;
    justify-content: flex-start !important;
    gap: 0.45rem 0.55rem !important;
}
[data-testid="stElementContainer"]:has(.chip-anchor) ~ [data-testid="stLayoutWrapper"] [data-testid="stHorizontalBlock"] > div {
    flex: 0 0 auto !important;
    flex-shrink: 0 !important;
    width: auto !important;
    min-width: max-content !important;
    max-width: none !important;
}
[data-testid="stElementContainer"]:has(.chip-anchor) ~ [data-testid="stLayoutWrapper"] [data-testid="stButton"] {
    width: auto !important;
    flex: 0 0 auto !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stElementContainer"]:has(.chip-anchor) ~ [data-testid="stLayoutWrapper"] [data-testid="stButton"] button,
[data-testid="stElementContainer"]:has(.chip-anchor) ~ [data-testid="stLayoutWrapper"] button {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    border-radius: 999px !important;
    background: #eeeeee !important;
    color: #374151 !important;
    min-height: 32px !important;
    height: 32px !important;
    min-width: unset !important;
    width: max-content !important;
    max-width: none !important;
    padding: 0 0.9rem 0 0.55rem !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    flex-shrink: 0 !important;
    overflow: visible !important;
}
[data-testid="stElementContainer"]:has(.chip-anchor) ~ [data-testid="stLayoutWrapper"] button:hover,
[data-testid="stElementContainer"]:has(.chip-anchor) ~ [data-testid="stLayoutWrapper"] button:focus,
[data-testid="stElementContainer"]:has(.chip-anchor) ~ [data-testid="stLayoutWrapper"] button:active {
    background: #e4e4e7 !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    color: #1f2937 !important;
}
[data-testid="stElementContainer"]:has(.chip-anchor) ~ [data-testid="stLayoutWrapper"] button p {
    font-size: 0.82rem !important;
    overflow: visible !important;
    text-overflow: unset !important;
    white-space: nowrap !important;
    width: max-content !important;
    max-width: none !important;
    flex: 0 0 auto !important;
}

.attach-close-mark + div button {
    border: none !important;
    background: #f3f4f6 !important;
    box-shadow: none !important;
    min-height: 36px !important;
    height: 36px !important;
    width: 36px !important;
    padding: 0 !important;
    border-radius: 999px !important;
    color: #4b5563 !important;
}
.attach-close-mark + div button:hover {
    background: #e5e7eb !important;
    color: #111827 !important;
}
.attach-row { margin: 0.35rem 0 0.15rem 0.15rem; }
.attach-chip {
    display: inline-block;
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 999px;
    padding: 0.18rem 0.7rem;
    font-size: 0.8rem;
    color: #374151;
}
.chat-attach {
    color: #1a7f7f;
    font-size: 0.8rem;
    margin: 0.25rem 0 0 0;
}

.audit-anchor + div button {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    min-height: 28px !important;
    height: 28px !important;
    width: 28px !important;
    padding: 0 !important;
    color: #6b7280 !important;
}
.audit-anchor + div button:hover {
    background: transparent !important;
    color: #1a7f7f !important;
}
.chat-time {
    color: #6b7280;
    font-size: 0.75rem;
    margin: 0.15rem 0 0 0;
}
</style>
"""


def _clock(moment: datetime | None) -> str:
    if moment is None:
        return ""
    return moment.strftime("%I:%M %p").lstrip("0")


def _as_message(item) -> dict:
    if isinstance(item, dict):
        return item
    role, text = item
    return {"role": role, "text": text, "sent_at": None, "received_at": None, "duration_s": None, "trace": None}


def _activity_for(question: str) -> tuple[str, list[str]]:
    """Pick a live status title and step lines from the wording of the question."""
    text = (question or "").lower()
    steps: list[str] = ["Planning the steps for this request…"]

    if any(word in text for word in ("register", "add my", "new member", "family member")):
        steps.append("Preparing to register a family member…")
    if any(word in text for word in ("histor", "summar", "chart", "medical record")):
        steps.append("Looking into the patient record and medical history…")
    if any(word in text for word in ("allerg",)):
        steps.append("Checking allergy flags on the record…")
    if any(word in text for word in ("note", "record that", "update", "reported")):
        steps.append("Preparing to update the patient record…")
    if any(word in text for word in ("book", "slot", "appoint", "schedule", "available", "next week")):
        steps.append("Looking up appointment slots…")
    if any(word in text for word in ("specialist", "nephrolog", "cardiolog", "doctor", "who can")):
        steps.append("Searching for a matching specialist…")
    if any(word in text for word in ("treatment", "condition", "disease", "ckd", "kidney", "latest")):
        steps.append("Searching trusted medical information…")
    if any(word in text for word in ("attached", "imported from", "shared ", "### ")):
        steps.append("Reading the attached document…")
    if len(steps) == 1:
        steps.append("Choosing tools and gathering an answer…")

    return steps[1], steps


def _queue_user(question: str, attachments: list[str] | None = None) -> None:
    st.session_state["chat"].append(
        {
            "role": "user",
            "text": question,
            "attachments": attachments or [],
            "sent_at": datetime.now(),
            "trace": None,
        }
    )


def _complete(question: str, sent_at: datetime | None) -> None:
    """Run the agent after the existing thread has already been drawn."""
    memory = get_memory()
    title, steps = _activity_for(question)
    with st.chat_message("assistant"):
        with st.status(title, expanded=True) as status:
            for step in steps:
                st.write(step)
            trace = get_agent().run(question, use_planner=True)
            seconds = max(trace.duration_ms / 1000, 0)
            status.update(label=f"Finished in {seconds:.0f}s", state="complete", expanded=False)

    record_trace(trace)
    log_run(trace, session_id=memory.thread_id)
    st.session_state["chat"].append(
        {
            "role": "assistant",
            "text": trace.answer or f"I could not complete that. {trace.error or ''}".strip(),
            "sent_at": sent_at,
            "received_at": datetime.now(),
            "duration_s": trace.duration_ms / 1000,
            "trace": trace,
        }
    )
    persist_assistant_chat()
    st.rerun()


def _queue_question(question: str) -> None:
    role = "guest" if is_guest() else "attendant"
    attendant_id = None if is_guest() else st.session_state.get("attendant_id")
    turn = build_turn(
        question,
        role=role,
        attendant_id=attendant_id,
        files_key=ATTENDANT_FILES,
        pending_key=ATTENDANT_PENDING_IMPORT,
    )
    if turn is None:
        return
    st.session_state[COMPOSER_KEY] = ""
    st.session_state["pending_question"] = turn.display
    st.session_state["pending_agent_prompt"] = turn.agent_prompt
    st.session_state["pending_attachments"] = turn.attachments
    st.rerun()


def _render_message(index: int, item) -> None:
    message = _as_message(item)
    role = message["role"]
    with st.chat_message(role):
        st.markdown(message["text"])
        if role == "user" and message.get("attachments"):
            st.markdown(
                '<p class="chat-attach">Attached: '
                + ", ".join(message["attachments"])
                + "</p>",
                unsafe_allow_html=True,
            )
        if role == "user" and message.get("sent_at"):
            st.markdown(
                f'<p class="chat-time">Sent {_clock(message["sent_at"])}</p>',
                unsafe_allow_html=True,
            )
            return

        parts = []
        if message.get("received_at"):
            parts.append(f"Received {_clock(message['received_at'])}")
        if message.get("duration_s") is not None:
            parts.append(f"{message['duration_s']:.0f}s")

        if message.get("trace") is None:
            if parts:
                st.markdown(
                    f'<p class="chat-time">{" · ".join(parts)}</p>',
                    unsafe_allow_html=True,
                )
            return

        open_key = f"audit_open_{index}"
        st.session_state.setdefault(open_key, False)
        time_col, audit_col = st.columns([9.2, 0.8], vertical_alignment="center")
        if parts:
            time_col.markdown(
                f'<p class="chat-time">{" · ".join(parts)}</p>',
                unsafe_allow_html=True,
            )
        with audit_col:
            st.markdown('<div class="audit-anchor"></div>', unsafe_allow_html=True)
            clicked = st.button(
                "",
                icon=":material/policy:",
                type="tertiary",
                key=f"audit_btn_{index}",
            )
        if clicked:
            st.session_state[open_key] = not st.session_state[open_key]
            st.rerun()
        if st.session_state[open_key]:
            render_trace(message["trace"], show_answer=False, expanded=False)


def render() -> None:
    guest = is_guest()
    page_header(
        settings.app_title,
        "Type a question, or tap a suggestion to send it.",
    )
    llm_ready = llm_warning()
    st.session_state.setdefault(COMPOSER_KEY, "")

    pending = st.session_state.pop("pending_question", None)
    agent_prompt = st.session_state.pop("pending_agent_prompt", None) or pending
    attachments = st.session_state.pop("pending_attachments", None)
    if pending and llm_ready:
        last = st.session_state["chat"][-1] if st.session_state["chat"] else None
        if not (isinstance(last, dict) and last.get("role") == "user" and last.get("text") == pending):
            _queue_user(pending, attachments)

    if guest:
        st.info(
            "You are browsing as a guest. You can look up doctors, open times, "
            "and general medical information, and attach a file to ask about it. "
            "This chat is not saved, and a new assistant can only be started after "
            "you sign in. Booking or filing a patient record also needs a sign-in.",
            icon=":material/person:",
        )
    elif not st.session_state["chat"] and not pending:
        st.info(
            "Ask me to book an appointment, summarise a relative's history, or "
            "register a family member. Attach a PDF, Word file, or image to ask "
            "about it — medical records are added to the matching chart, or I "
            "will ask for the name if it is not clear.",
            icon=":material/waving_hand:",
        )

    for index, item in enumerate(st.session_state["chat"]):
        _render_message(index, item)

    live = st.container()
    busy = bool(pending and llm_ready)

    st.markdown(COMPOSER_CSS, unsafe_allow_html=True)
    st.markdown('<div class="composer-anchor"></div>', unsafe_allow_html=True)
    attach_col, field_col = st.columns([0.55, 9.45], vertical_alignment="center")
    with attach_col:
        render_attach_button(ATTENDANT_FILES, disabled=not llm_ready or busy)
    with field_col, st.form("assistant_composer", clear_on_submit=True, border=False):
        input_col, send_col = st.columns([8.9, 0.55], vertical_alignment="center")
        draft = input_col.text_input(
            "Your message",
            placeholder="Ask, or attach a file",
            disabled=not llm_ready or busy,
            label_visibility="collapsed",
        )
        sent = send_col.form_submit_button(
            "",
            icon=":material/arrow_upward:",
            help="Send",
            type="primary",
            width="stretch",
            disabled=not llm_ready or busy,
        )
    if sent:
        _queue_question(draft)
    render_attach_picker(ATTENDANT_FILES, disabled=not llm_ready or busy)
    render_file_chips(ATTENDANT_FILES)

    st.markdown('<div class="chip-anchor"></div>', unsafe_allow_html=True)
    replies = GUEST_REPLIES if guest else QUICK_REPLIES
    with st.container(horizontal=True, wrap=True, gap="small"):
        for label in replies:
            if st.button(
                label,
                icon=":material/link:",
                type="tertiary",
                width="content",
                key=f"chip_{label}",
                disabled=not llm_ready or busy,
            ):
                st.session_state[COMPOSER_KEY] = replies[label]
                st.session_state["pending_question"] = replies[label]
                st.rerun()

    if busy:
        sent_at = None
        last = st.session_state["chat"][-1] if st.session_state["chat"] else None
        if isinstance(last, dict):
            sent_at = last.get("sent_at")
        with live:
            _complete(agent_prompt, sent_at)
