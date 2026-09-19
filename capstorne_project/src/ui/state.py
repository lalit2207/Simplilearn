"""Shared Streamlit session state and cached resources.

Streamlit re-runs the entire script on every interaction, so anything
expensive or stateful has to be held deliberately:

  * `@st.cache_resource` for the embedding model - loading it per rerun would
    add seconds to every click.
  * `st.session_state` for the agent session, so conversation memory and the
    trace history survive reruns.
"""

from __future__ import annotations

import streamlit as st

from src.agent import (
    HealthcareAgent,
    build_doctor_agent,
    build_doctor_session,
    build_guest_agent,
    build_guest_session,
    build_session,
)
from src.data import is_seeded, seed_database
from src.data.chat_store import (
    create_session,
    get_session,
    latest_session,
    list_sessions,
    load_messages,
    save_session,
    title_from_messages,
)
from src.data.database import fetch_all, init_db
from src.rag.embeddings import get_embeddings
from src.ui.composer_files import ATTENDANT_FILES, DOCTOR_FILES, clear_files

ROLES = ("Attendant", "Doctor", "Admin / LLMOps")

# Admin continues as a generic admin user. Attendant and Doctor pick who
# they are on the next page (dropdowns stand in for a login id later).
ROLE_USERS = {
    "Attendant": {"attendant_id": 1, "display_name": "Lalit Chauhan"},
    "Doctor": {"display_name": "Doctor"},
    "Admin / LLMOps": {"display_name": "Admin user"},
}


@st.cache_resource(show_spinner="Loading embedding model...")
def warm_embeddings():
    """Load the sentence-transformers model once per process."""
    return get_embeddings()


@st.cache_resource(show_spinner="Preparing database...")
def ensure_database() -> bool:
    """Create missing tables (including chat history) and seed on first launch."""
    init_db()
    if not is_seeded():
        seed_database()
    return True


@st.cache_data(ttl=30)
def list_attendants() -> list[dict]:
    return fetch_all("SELECT id, full_name, email FROM users ORDER BY id")


@st.cache_data(ttl=30)
def list_doctors() -> list[dict]:
    return fetch_all(
        "SELECT id, full_name, specialty, hospital FROM doctors ORDER BY specialty, full_name"
    )


def init_state() -> None:
    defaults = {
        # None means "no role chosen yet", which is what shows the landing screen.
        "role": None,
        "attendant_access": None,  # None | guest | registered
        "doctor_access": None,  # None | selected
        "attendant_id": 1,
        "doctor_id": 1,
        "traces": [],          # AgentTrace objects from this session
        "chat": [],            # attendant chat transcript
        "doctor_chat": [],     # doctor assistant transcript
        "last_search": None,   # most recent medical search results
        "pending_question": None,  # queued by a quick-reply chip
        "doctor_pending_question": None,
        "assistant_session_id": None,
        "doctor_assistant_session_id": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def current_role() -> str | None:
    return st.session_state.get("role")


def set_role(role: str) -> None:
    """Pick a role. Attendant and Doctor still choose an identity on the next page."""
    identity = ROLE_USERS.get(role, {})
    st.session_state["role"] = role
    st.session_state["attendant_access"] = None
    st.session_state["doctor_access"] = None
    if role in {"Attendant", "Doctor"}:
        st.session_state["signed_in_name"] = None
        return
    st.session_state["signed_in_name"] = identity.get("display_name", role)


def continue_as_guest() -> None:
    st.session_state["attendant_access"] = "guest"
    st.session_state["attendant_id"] = 0
    st.session_state["signed_in_name"] = "Guest"
    st.session_state["memory"] = None
    st.session_state["chat"] = []
    st.session_state["pending_question"] = None
    st.session_state["pending_record_import"] = None
    st.session_state["assistant_session_id"] = None
    clear_files(ATTENDANT_FILES)


def continue_as_registered(attendant_id: int, display_name: str) -> None:
    st.session_state["attendant_access"] = "registered"
    st.session_state["attendant_id"] = attendant_id
    st.session_state["signed_in_name"] = display_name
    st.session_state["memory"] = None
    st.session_state["chat"] = []
    st.session_state["pending_question"] = None
    st.session_state["pending_record_import"] = None
    st.session_state["assistant_session_id"] = None
    clear_files(ATTENDANT_FILES)


def reset_attendant_access(*, open_signup: bool = False) -> None:
    """Return to Guest / registered / sign-up without leaving the Attendant role."""
    st.session_state["attendant_access"] = None
    st.session_state["signed_in_name"] = None
    st.session_state["memory"] = None
    st.session_state["chat"] = []
    st.session_state["pending_question"] = None
    st.session_state["pending_record_import"] = None
    st.session_state["assistant_session_id"] = None
    clear_files(ATTENDANT_FILES)
    st.session_state["open_signup"] = open_signup


def is_guest() -> bool:
    return current_role() == "Attendant" and st.session_state.get("attendant_access") == "guest"


def attendant_ready() -> bool:
    return current_role() != "Attendant" or st.session_state.get("attendant_access") in {
        "guest",
        "registered",
    }


def continue_as_doctor(doctor_id: int, display_name: str) -> None:
    st.session_state["doctor_access"] = "selected"
    st.session_state["doctor_id"] = doctor_id
    st.session_state["signed_in_name"] = display_name
    st.session_state["doctor_memory"] = None
    st.session_state["doctor_chat"] = []
    st.session_state["doctor_pending_question"] = None
    st.session_state["doctor_pending_record_import"] = None
    st.session_state["doctor_assistant_session_id"] = None
    clear_files(DOCTOR_FILES)


def reset_doctor_access() -> None:
    """Return to the doctor picker without leaving the Doctor role."""
    st.session_state["doctor_access"] = None
    st.session_state["signed_in_name"] = None
    st.session_state["doctor_memory"] = None
    st.session_state["doctor_chat"] = []
    st.session_state["doctor_pending_question"] = None
    st.session_state["doctor_pending_record_import"] = None
    st.session_state["doctor_assistant_session_id"] = None
    clear_files(DOCTOR_FILES)


def doctor_ready() -> bool:
    return current_role() != "Doctor" or st.session_state.get("doctor_access") == "selected"


def clear_role() -> None:
    """Return to the landing screen without discarding session data."""
    st.session_state["role"] = None
    st.session_state["attendant_access"] = None
    st.session_state["doctor_access"] = None


def current_user_name() -> str:
    """The person logged in when the role was chosen."""
    name = st.session_state.get("signed_in_name")
    if name:
        return name
    if is_guest():
        return "Guest"
    role = current_role()
    return ROLE_USERS.get(role or "", {}).get("display_name", role or "Guest")


def initials(name: str) -> str:
    parts = [part for part in (name or "").split() if part]
    return "".join(part[0].upper() for part in parts[:2]) or "?"


def assistant_owner() -> tuple[str, int] | None:
    """Registered attendant or signed-in doctor who may persist chats. Guests: None."""
    if current_role() == "Attendant" and st.session_state.get("attendant_access") == "registered":
        return ("attendant", int(st.session_state.get("attendant_id") or 0))
    if current_role() == "Doctor" and st.session_state.get("doctor_access") == "selected":
        return ("doctor", int(st.session_state.get("doctor_id") or 0))
    return None


def can_start_new_assistant() -> bool:
    return assistant_owner() is not None


def _session_id_key(owner_type: str) -> str:
    return "doctor_assistant_session_id" if owner_type == "doctor" else "assistant_session_id"


def _chat_key(owner_type: str) -> str:
    return "doctor_chat" if owner_type == "doctor" else "chat"


def _memory_key(owner_type: str) -> str:
    return "doctor_memory" if owner_type == "doctor" else "memory"


def _restore_turns(memory, chat: list) -> None:
    memory.turns = []
    pending_q = None
    for item in chat:
        msg = item if isinstance(item, dict) else {"role": item[0], "text": item[1]}
        if msg.get("role") == "user":
            pending_q = msg.get("text") or ""
        elif msg.get("role") == "assistant" and pending_q is not None:
            memory.record(pending_q, msg.get("text") or "")
            pending_q = None


def persist_assistant_chat() -> None:
    """Write the live transcript for a registered user. Guests are skipped."""
    owner = assistant_owner()
    if owner is None:
        return
    owner_type, owner_id = owner
    chat = st.session_state.get(_chat_key(owner_type)) or []
    sid = st.session_state.get(_session_id_key(owner_type))
    memory = st.session_state.get(_memory_key(owner_type))
    thread_id = getattr(memory, "thread_id", None)
    if not sid:
        if not chat:
            return
        sid = create_session(owner_type, owner_id, thread_id=thread_id)
        st.session_state[_session_id_key(owner_type)] = sid
    save_session(
        sid,
        chat,
        title=title_from_messages(chat),
        thread_id=thread_id,
    )


def load_assistant_session(session_id: int) -> bool:
    """Replace the live chat with a stored session owned by the current user."""
    owner = assistant_owner()
    if owner is None:
        return False
    owner_type, owner_id = owner
    row = get_session(session_id)
    if not row or row["owner_type"] != owner_type or int(row["owner_id"]) != owner_id:
        return False
    chat = load_messages(session_id)
    if owner_type == "doctor":
        doctors = {item["id"]: item["full_name"] for item in list_doctors()}
        memory = build_doctor_session(
            doctor_id=owner_id,
            doctor_name=doctors.get(owner_id, f"Doctor {owner_id}"),
            thread_id=row.get("thread_id"),
        )
        st.session_state["doctor_memory"] = memory
        st.session_state["doctor_chat"] = chat
        st.session_state["doctor_assistant_session_id"] = session_id
        st.session_state["doctor_pending_question"] = None
        st.session_state["doctor_pending_agent_prompt"] = None
        st.session_state["doctor_pending_attachments"] = None
        st.session_state["doctor_pending_record_import"] = None
        clear_files(DOCTOR_FILES)
    else:
        attendants = {item["id"]: item["full_name"] for item in list_attendants()}
        memory = build_session(
            attendant_id=owner_id,
            attendant_name=attendants.get(owner_id, f"Attendant {owner_id}"),
            thread_id=row.get("thread_id"),
        )
        st.session_state["memory"] = memory
        st.session_state["chat"] = chat
        st.session_state["assistant_session_id"] = session_id
        st.session_state["pending_question"] = None
        st.session_state["pending_agent_prompt"] = None
        st.session_state["pending_attachments"] = None
        st.session_state["pending_record_import"] = None
        clear_files(ATTENDANT_FILES)
    _restore_turns(memory, chat)
    return True


def open_assistant_session(session_id: int) -> bool:
    persist_assistant_chat()
    return load_assistant_session(session_id)


def list_saved_assistant_sessions() -> list[dict]:
    owner = assistant_owner()
    if owner is None:
        return []
    owner_type, owner_id = owner
    return list_sessions(owner_type, owner_id)


def current_assistant_session_id() -> int | None:
    owner = assistant_owner()
    if owner is None:
        return None
    return st.session_state.get(_session_id_key(owner[0]))


def ensure_assistant_session() -> None:
    """Resume the latest saved thread for a registered user, if any."""
    owner = assistant_owner()
    if owner is None:
        return
    owner_type, owner_id = owner
    if st.session_state.get(_session_id_key(owner_type)):
        return
    # Do not replace an in-flight transcript that has not been saved yet.
    if st.session_state.get(_chat_key(owner_type)):
        persist_assistant_chat()
        return
    latest = latest_session(owner_type, owner_id)
    if latest:
        load_assistant_session(latest["id"])


def get_memory():
    """The agent's session memory, rebuilt if the attendant or guest mode changes.

    Switching attendant must start a fresh thread - otherwise one caregiver's
    conversation context would carry into another's session.
    """
    memory = st.session_state.get("memory")
    if is_guest():
        if memory is None or memory.role != "guest":
            switching = memory is not None
            memory = build_guest_session()
            st.session_state["memory"] = memory
            # First-time create must not wipe a question already queued in chat.
            if switching:
                st.session_state["chat"] = []
        return memory

    attendant_id = st.session_state["attendant_id"]
    if memory is None or memory.attendant_id != attendant_id or memory.role == "guest":
        switching = memory is not None
        attendants = {a["id"]: a["full_name"] for a in list_attendants()}
        memory = build_session(
            attendant_id=attendant_id,
            attendant_name=attendants.get(attendant_id, f"Attendant {attendant_id}"),
        )
        st.session_state["memory"] = memory
        if switching:
            st.session_state["chat"] = []
            st.session_state["assistant_session_id"] = None
    ensure_assistant_session()
    return st.session_state.get("memory") or memory


def start_new_thread() -> None:
    """Open a new saved assistant session. Guests cannot start another thread."""
    if is_guest() or assistant_owner() is None:
        return
    persist_assistant_chat()
    if st.session_state.get("assistant_session_id") and not st.session_state.get("chat"):
        return
    attendant_id = st.session_state.get("attendant_id", 1)
    attendants = {a["id"]: a["full_name"] for a in list_attendants()}
    memory = build_session(
        attendant_id=attendant_id,
        attendant_name=attendants.get(attendant_id, f"Attendant {attendant_id}"),
    )
    st.session_state["memory"] = memory
    st.session_state["assistant_session_id"] = create_session(
        "attendant", attendant_id, thread_id=memory.thread_id
    )
    st.session_state["chat"] = []
    st.session_state["pending_question"] = None
    st.session_state["pending_agent_prompt"] = None
    st.session_state["pending_attachments"] = None
    st.session_state["pending_record_import"] = None
    clear_files(ATTENDANT_FILES)
    st.session_state["assistant_draft"] = ""


def get_agent() -> HealthcareAgent:
    if is_guest():
        return build_guest_agent(get_memory())
    return HealthcareAgent(get_memory())


def get_doctor_memory():
    """Doctor-assistant session, rebuilt if the signed-in doctor changes."""
    doctor_id = st.session_state.get("doctor_id", 1)
    memory = st.session_state.get("doctor_memory")
    if memory is None or memory.doctor_id != doctor_id:
        switching = memory is not None
        doctors = {row["id"]: row["full_name"] for row in list_doctors()}
        memory = build_doctor_session(
            doctor_id=doctor_id,
            doctor_name=doctors.get(doctor_id, f"Doctor {doctor_id}"),
        )
        st.session_state["doctor_memory"] = memory
        if switching:
            st.session_state["doctor_chat"] = []
            st.session_state["doctor_assistant_session_id"] = None
    ensure_assistant_session()
    return st.session_state.get("doctor_memory") or memory


def start_new_doctor_thread() -> None:
    if assistant_owner() is None or assistant_owner()[0] != "doctor":
        return
    persist_assistant_chat()
    if st.session_state.get("doctor_assistant_session_id") and not st.session_state.get(
        "doctor_chat"
    ):
        return
    doctor_id = st.session_state.get("doctor_id", 1)
    doctors = {row["id"]: row["full_name"] for row in list_doctors()}
    memory = build_doctor_session(
        doctor_id=doctor_id,
        doctor_name=doctors.get(doctor_id, f"Doctor {doctor_id}"),
    )
    st.session_state["doctor_memory"] = memory
    st.session_state["doctor_assistant_session_id"] = create_session(
        "doctor", doctor_id, thread_id=memory.thread_id
    )
    st.session_state["doctor_chat"] = []
    st.session_state["doctor_pending_question"] = None
    st.session_state["doctor_pending_agent_prompt"] = None
    st.session_state["doctor_pending_attachments"] = None
    st.session_state["doctor_pending_record_import"] = None
    clear_files(DOCTOR_FILES)
    st.session_state["doctor_draft"] = ""


def get_doctor_agent() -> HealthcareAgent:
    return build_doctor_agent(get_doctor_memory())


def record_trace(trace) -> None:
    st.session_state["traces"].append(trace)


def session_traces() -> list:
    return st.session_state.get("traces", [])
