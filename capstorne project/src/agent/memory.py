"""Memory modules.

The capstone asks for memory that "retains long-term patient context". That
splits into two different mechanisms, and conflating them is a common mistake:

  * Short-term  - the running conversation, so "book it for him too" resolves.
                  Held per session in a LangGraph checkpointer, windowed to the
                  last MEMORY_WINDOW turns so the context stays bounded.
  * Long-term   - durable facts about a patient. These live in SQLite (the
                  chart) and FAISS (embedded chunks), so they survive restarts
                  and are retrieved by relevance rather than recency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from langgraph.checkpoint.memory import InMemorySaver

from src.config import settings
from src.rag.indexer import retrieve_patient_context
from src.tools.appointment_tools import format_doctor_roster
from src.tools.patient_tools import format_family_roster, get_active_alerts, get_patient_profile

# One checkpointer per process. LangGraph keys conversations by thread_id, so
# every attendant session gets its own isolated history.
_checkpointer = InMemorySaver()


def get_checkpointer() -> InMemorySaver:
    return _checkpointer


@dataclass
class ConversationTurn:
    question: str
    answer: str
    patient_id: int | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class SessionMemory:
    """Short-term memory for one attendant session.

    This mirrors what the checkpointer holds, but in a form the Streamlit UI
    can render directly as a memory trace.
    """

    attendant_id: int
    attendant_name: str
    thread_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    last_patient_id: int | None = None
    role: str = "attendant"
    doctor_id: int | None = None
    doctor_name: str = ""

    def record(self, question: str, answer: str, patient_id: int | None = None) -> None:
        self.turns.append(ConversationTurn(question, answer, patient_id))
        if patient_id is not None:
            self.last_patient_id = patient_id
        # Keep only the most recent window so prompts do not grow unbounded.
        if len(self.turns) > settings.memory_window:
            self.turns = self.turns[-settings.memory_window :]

    def recent_dialogue(self, limit: int = 3) -> str:
        if not self.turns:
            return ""
        lines = []
        for turn in self.turns[-limit:]:
            answer = turn.answer if len(turn.answer) <= 300 else turn.answer[:300] + "..."
            speaker = "Doctor" if self.role == "doctor" else "Attendant"
            lines.append(f"{speaker}: {turn.question}\nAssistant: {answer}")
        return "\n\n".join(lines)

    def clear(self) -> None:
        self.turns.clear()
        self.last_patient_id = None


def long_term_patient_context(patient_id: int, query: str, k: int | None = None) -> str:
    """Durable patient context, retrieved by relevance to the current query.

    This is what makes the memory *long-term*: it reads from FAISS and SQLite,
    not from the conversation, so it works on the first turn of a brand-new
    session and after a restart.
    """
    profile = get_patient_profile(patient_id)
    if not profile:
        return ""

    lines = [
        f"Known patient: {profile['full_name']} (id {patient_id}), "
        f"age {profile.get('age')}, allergies: {profile.get('allergies') or 'none recorded'}."
    ]

    # Alerts are per-record, so one long-running condition appears many times.
    # Keep the highest severity per condition to avoid padding the prompt.
    highest: dict[str, str] = {}
    for alert in get_active_alerts(patient_id):
        highest.setdefault(alert["condition"] or "Unspecified", alert["alert_level"])
    if highest:
        lines.append(
            "Standing alerts: "
            + "; ".join(f"{condition} ({level})" for condition, level in highest.items())
        )

    hits = retrieve_patient_context(patient_id, query, k=k)
    if hits:
        lines.append("Relevant history retrieved for this query:")
        lines.extend(f"  - {hit['text']}" for hit in hits)

    return "\n".join(lines)


def build_memory_section(memory: SessionMemory, query: str, patient_id: int | None) -> str:
    """Assemble the memory block injected into the agent prompt."""
    if memory.role == "doctor" and memory.doctor_id is not None:
        blocks: list[str] = [format_doctor_roster(memory.doctor_id)]
    elif memory.role == "guest":
        blocks = [
            "Visitor is a guest and is not signed in. No family roster or "
            "patient chart is available until they sign in as a registered user."
        ]
    else:
        blocks = [format_family_roster(memory.attendant_id)]

    dialogue = memory.recent_dialogue()
    if dialogue:
        blocks.append(f"Recent conversation (short-term memory):\n{dialogue}")

    focus_patient = patient_id or memory.last_patient_id
    if focus_patient:
        context = long_term_patient_context(focus_patient, query)
        if context:
            blocks.append(f"Patient context (long-term memory):\n{context}")

    if not blocks:
        return "No prior context for this session.\n"
    return "\n\n".join(blocks) + "\n"
