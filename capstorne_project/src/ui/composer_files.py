"""Shared attach control for every assistant composer."""

from __future__ import annotations

import streamlit as st

from src.attachments.ingest import PreparedTurn, prepare_turn

ALLOWED_TYPES = ["pdf", "doc", "docx", "png", "jpg", "jpeg", "webp"]
MAX_FILES = 5
MAX_BYTES = 8 * 1024 * 1024

ATTENDANT_FILES = "assistant_files"
DOCTOR_FILES = "doctor_files"
ATTENDANT_PENDING_IMPORT = "pending_record_import"
DOCTOR_PENDING_IMPORT = "doctor_pending_record_import"


def render_attach_button(files_key: str, *, disabled: bool) -> None:
    """Paperclip inside the composer pill. Opens the file picker when clicked."""
    st.session_state.setdefault(files_key, [])
    open_key = f"{files_key}_open"
    st.session_state.setdefault(open_key, False)
    if st.button(
        "",
        icon=":material/attach_file:",
        help="Attach PDF, Word, or image",
        width="stretch",
        disabled=disabled,
        key=f"{files_key}_clip",
    ):
        st.session_state[open_key] = not st.session_state[open_key]
        st.rerun()


def render_attach_picker(files_key: str, *, disabled: bool) -> None:
    """File picker shown under the composer after the paperclip is tapped."""
    if not st.session_state.get(f"{files_key}_open"):
        return
    uploads = None
    pick_col, close_col = st.columns([9.35, 0.65], vertical_alignment="center")
    with pick_col:
        generation = st.session_state.get(f"{files_key}_gen", 0)
        uploads = st.file_uploader(
            "PDF, Word, or image",
            type=ALLOWED_TYPES,
            accept_multiple_files=True,
            key=f"{files_key}_uploader_{generation}",
            disabled=disabled,
            label_visibility="collapsed",
        )
    with close_col:
        st.markdown('<div class="attach-close-mark"></div>', unsafe_allow_html=True)
        if st.button(
            "",
            icon=":material/close:",
            help="Close",
            type="tertiary",
            width="stretch",
            key=f"{files_key}_close",
        ):
            st.session_state[f"{files_key}_open"] = False
            st.rerun()
    if uploads:
        _stash(files_key, uploads)
        st.session_state[f"{files_key}_open"] = False
        st.session_state[f"{files_key}_gen"] = generation + 1
        st.rerun()


def render_file_chips(files_key: str) -> None:
    files = list(st.session_state.get(files_key) or [])
    if not files:
        return
    st.markdown('<div class="attach-row">', unsafe_allow_html=True)
    for index, item in enumerate(files):
        label = item["name"]
        col_name, col_remove = st.columns([8.6, 1.4], vertical_alignment="center")
        col_name.markdown(f'<span class="attach-chip">{_escape(label)}</span>', unsafe_allow_html=True)
        if col_remove.button(
            "Remove",
            key=f"{files_key}_rm_{index}_{label}",
            type="tertiary",
        ):
            files.pop(index)
            st.session_state[files_key] = files
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def take_files(files_key: str) -> list[tuple[str, bytes, str | None]]:
    stored = list(st.session_state.get(files_key) or [])
    clear_files(files_key)
    return [(item["name"], item["data"], item.get("type")) for item in stored]


def clear_files(files_key: str) -> None:
    st.session_state[files_key] = []
    st.session_state[f"{files_key}_open"] = False
    st.session_state[f"{files_key}_gen"] = st.session_state.get(f"{files_key}_gen", 0) + 1


def build_turn(
    question: str,
    *,
    role: str,
    attendant_id: int | None,
    files_key: str,
    pending_key: str,
) -> PreparedTurn | None:
    typed = (question or "").strip()
    stored = list(st.session_state.get(files_key) or [])
    if not typed and not stored:
        return None
    files = take_files(files_key)
    turn = prepare_turn(
        question,
        files,
        role=role,
        attendant_id=attendant_id,
        pending_import=st.session_state.get(pending_key),
    )
    st.session_state[pending_key] = turn.pending_import
    return turn


def _stash(files_key: str, uploads) -> None:
    existing = {item["name"]: item for item in st.session_state.get(files_key) or []}
    for upload in uploads:
        data = upload.getvalue()
        if len(data) > MAX_BYTES:
            st.warning(f"{upload.name} is larger than 8 MB and was skipped.")
            continue
        existing[upload.name] = {
            "name": upload.name,
            "type": getattr(upload, "type", None),
            "data": data,
        }
    st.session_state[files_key] = list(existing.values())[:MAX_FILES]


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
