"""Doctor assistant: query patients, charts, and the clinic list."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from src.config import settings
from src.llmops import log_run
from src.ui.components import llm_warning, page_header, render_trace
from src.ui.composer_files import (
    DOCTOR_FILES,
    DOCTOR_PENDING_IMPORT,
    build_turn,
    render_attach_button,
    render_attach_picker,
    render_file_chips,
)
from src.ui.state import get_doctor_agent, get_doctor_memory, persist_assistant_chat, record_trace
from src.ui.views.assistant import COMPOSER_CSS, _as_message, _clock

QUICK_REPLIES = {
    "Summarise Ramesh": "Summarise Ramesh Chauhan's medical history and flag allergies.",
    "Sunita allergies": "Does Sunita Chauhan have any allergies I should know before prescribing?",
    "Who is on my list": "Who is on my confirmed appointment list?",
    "CKD treatments": "What are current treatment approaches for chronic kidney disease?",
}

COMPOSER_KEY = "doctor_draft"
PENDING_KEY = "doctor_pending_question"
CHAT_KEY = "doctor_chat"


def _activity_for(question: str) -> tuple[str, list[str]]:
    text = (question or "").lower()
    steps = ["Planning the steps for this request…"]
    if any(word in text for word in ("histor", "summar", "chart", "record", "allerg")):
        steps.append("Looking up the patient and opening the chart…")
    if any(word in text for word in ("list", "today", "appointment", "who is", "who are")):
        steps.append("Checking your clinic appointments…")
    if any(word in text for word in ("treatment", "condition", "disease", "ckd", "kidney")):
        steps.append("Searching trusted medical information…")
    if any(word in text for word in ("attached", "imported from", "shared ", "### ")):
        steps.append("Reading the attached document…")
    if len(steps) == 1:
        steps.append("Choosing tools and gathering an answer…")
    return steps[1], steps


def _queue_user(question: str, attachments: list[str] | None = None) -> None:
    st.session_state[CHAT_KEY].append(
        {
            "role": "user",
            "text": question,
            "attachments": attachments or [],
            "sent_at": datetime.now(),
            "trace": None,
        }
    )


def _complete(question: str, sent_at: datetime | None) -> None:
    memory = get_doctor_memory()
    title, steps = _activity_for(question)
    with st.chat_message("assistant"):
        with st.status(title, expanded=True) as status:
            for step in steps:
                st.write(step)
            trace = get_doctor_agent().run(question, use_planner=True)
            seconds = max(trace.duration_ms / 1000, 0)
            status.update(label=f"Finished in {seconds:.0f}s", state="complete", expanded=False)

    record_trace(trace)
    log_run(trace, session_id=memory.thread_id)
    st.session_state[CHAT_KEY].append(
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
    turn = build_turn(
        question,
        role="doctor",
        attendant_id=None,
        files_key=DOCTOR_FILES,
        pending_key=DOCTOR_PENDING_IMPORT,
    )
    if turn is None:
        return
    st.session_state[COMPOSER_KEY] = ""
    st.session_state[PENDING_KEY] = turn.display
    st.session_state["doctor_pending_agent_prompt"] = turn.agent_prompt
    st.session_state["doctor_pending_attachments"] = turn.attachments
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

        open_key = f"doctor_audit_open_{index}"
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
                key=f"doctor_audit_btn_{index}",
            )
        if clicked:
            st.session_state[open_key] = not st.session_state[open_key]
            st.rerun()
        if st.session_state[open_key]:
            render_trace(message["trace"], show_answer=False, expanded=False)


def render() -> None:
    page_header(
        settings.app_title,
        "Ask about a patient by name, review their chart, or check who is on your list.",
    )
    llm_ready = llm_warning()
    st.session_state.setdefault(COMPOSER_KEY, "")
    st.session_state.setdefault(CHAT_KEY, [])

    pending = st.session_state.pop(PENDING_KEY, None)
    agent_prompt = st.session_state.pop("doctor_pending_agent_prompt", None) or pending
    attachments = st.session_state.pop("doctor_pending_attachments", None)
    if pending and llm_ready:
        last = st.session_state[CHAT_KEY][-1] if st.session_state[CHAT_KEY] else None
        if not (isinstance(last, dict) and last.get("role") == "user" and last.get("text") == pending):
            _queue_user(pending, attachments)

    if not st.session_state[CHAT_KEY] and not pending:
        st.info(
            "Name a patient to open their history, ask who is booked with you, "
            "or look up a condition. Attach a PDF, Word file, or image — medical "
            "records are filed on the matching chart, or I will ask for the name.",
            icon=":material/stethoscope:",
        )

    for index, item in enumerate(st.session_state[CHAT_KEY]):
        _render_message(index, item)

    live = st.container()
    busy = bool(pending and llm_ready)

    st.markdown(COMPOSER_CSS, unsafe_allow_html=True)
    st.markdown('<div class="composer-anchor"></div>', unsafe_allow_html=True)
    attach_col, field_col = st.columns([0.55, 9.45], vertical_alignment="center")
    with attach_col:
        render_attach_button(DOCTOR_FILES, disabled=not llm_ready or busy)
    with field_col, st.form("doctor_composer", clear_on_submit=True, border=False):
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
    render_attach_picker(DOCTOR_FILES, disabled=not llm_ready or busy)
    render_file_chips(DOCTOR_FILES)

    st.markdown('<div class="chip-anchor"></div>', unsafe_allow_html=True)
    with st.container(horizontal=True, wrap=True, gap="small"):
        for label in QUICK_REPLIES:
            if st.button(
                label,
                icon=":material/link:",
                type="tertiary",
                width="content",
                key=f"doctor_chip_{label}",
                disabled=not llm_ready or busy,
            ):
                st.session_state[PENDING_KEY] = QUICK_REPLIES[label]
                st.rerun()

    if busy:
        sent_at = None
        last = st.session_state[CHAT_KEY][-1] if st.session_state[CHAT_KEY] else None
        if isinstance(last, dict):
            sent_at = last.get("sent_at")
        with live:
            _complete(agent_prompt, sent_at)
