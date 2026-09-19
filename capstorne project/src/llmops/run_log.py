"""Structured run logging.

The capstone asks to "log and analyze agent performance per module". That
requires machine-readable logs, so every run is appended as one JSON object
per line (JSONL) rather than as formatted text.

JSONL is chosen deliberately: appending is atomic enough for a single-process
app, a corrupt line never destroys the whole file, and pandas reads it
directly for the Streamlit metrics page.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from src.config import settings

RUN_LOG_NAME = "agent_runs.jsonl"
EVAL_LOG_NAME = "evaluations.jsonl"

# Which agent module each tool belongs to, so success rates can be reported
# per module the way the capstone asks rather than per individual tool.
TOOL_MODULES = {
    "resolve_patient": "patient_identification",
    "search_patients": "patient_identification",
    "list_family_members": "patient_identification",
    "list_doctor_appointments": "appointment_booking",
    "register_patient": "record_management",
    "get_patient_history": "history_retrieval",
    "add_patient_record": "record_management",
    "find_doctors": "appointment_booking",
    "find_available_slots": "appointment_booking",
    "book_appointment": "appointment_booking",
    "list_appointments": "appointment_booking",
    "cancel_appointment": "appointment_booking",
    "search_medical_information": "medical_search",
}


def module_for(tool_name: str) -> str:
    return TOOL_MODULES.get(tool_name, "other")


def _log_path(name: str) -> Path:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    return settings.log_dir / name


def append_jsonl(name: str, payload: dict[str, Any]) -> None:
    with _log_path(name).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def read_jsonl(name: str) -> list[dict]:
    """Read a log file, skipping any line that failed to serialise."""
    path = _log_path(name)
    if not path.exists():
        return []

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def log_run(trace, session_id: str = "", extra: dict | None = None) -> dict:
    """Persist one agent run, flattened for analysis.

    `trace` is an AgentTrace. Per-tool outcomes are recorded with their module
    so metrics can be grouped without re-deriving the mapping later.
    """
    payload = {
        "timestamp": trace.timestamp or datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "question": trace.question,
        "answer": trace.answer,
        "patient_id": trace.patient_id,
        "succeeded": trace.succeeded,
        "error": trace.error,
        "duration_ms": trace.duration_ms,
        "model": settings.groq_model,
        "plan_source": trace.plan.source if trace.plan else None,
        "planned_steps": len(trace.plan.steps) if trace.plan else 0,
        "planned_tools": trace.plan.tools_planned if trace.plan else [],
        "tools_called": trace.tools_used,
        "tool_calls": [
            {
                "name": call.name,
                "module": module_for(call.name),
                "arguments": call.arguments,
                "ok": call.ok,
                "output": call.output,
            }
            for call in trace.tool_calls
        ],
        "tool_failures": len(trace.failed_calls),
    }
    if extra:
        payload.update(extra)

    append_jsonl(RUN_LOG_NAME, payload)
    return payload


def log_evaluation(record: dict) -> dict:
    record.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    append_jsonl(EVAL_LOG_NAME, record)
    return record


def load_runs() -> list[dict]:
    return read_jsonl(RUN_LOG_NAME)


def load_evaluations() -> list[dict]:
    return read_jsonl(EVAL_LOG_NAME)


def iter_tool_calls(runs: list[dict] | None = None) -> Iterator[dict]:
    """Flatten every tool call across runs - the basis for module metrics."""
    for run in runs if runs is not None else load_runs():
        for call in run.get("tool_calls", []):
            yield {**call, "timestamp": run.get("timestamp"), "question": run.get("question")}


def clear_logs() -> None:
    for name in (RUN_LOG_NAME, EVAL_LOG_NAME):
        path = _log_path(name)
        if path.exists():
            path.unlink()
