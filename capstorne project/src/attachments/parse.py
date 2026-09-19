"""Decide whether extracted text is a medical record and pull out a patient name."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

RECORD_HINTS = (
    "diagnosis",
    "diagnosed",
    "prescription",
    "prescribed",
    "medication",
    "medications",
    "allergy",
    "allergies",
    "lab result",
    "laboratory",
    "investigation",
    "blood pressure",
    "medical history",
    "patient name",
    "patient:",
    "clinical",
    "impression",
    "discharge",
    "findings",
    "hemoglobin",
    "creatinine",
    "tablet",
    "rx ",
    "treatment",
    "condition",
    "chief complaint",
    "vital",
)

FILE_INTENT = (
    "add this",
    "file this",
    "update the record",
    "update the chart",
    "add to the record",
    "add to the chart",
    "medical record",
    "lab report",
    "prescription",
    "save this",
    "put this on",
)

NAME_STOP = {
    "diagnosis",
    "diagnosed",
    "medications",
    "medication",
    "treatment",
    "lab",
    "result",
    "allergies",
    "allergy",
    "condition",
    "notes",
    "date",
    "age",
    "gender",
    "male",
    "female",
    "blood",
    "history",
    "prescription",
}

NAME_PATTERNS = (
    re.compile(
        r"(?:patient(?:\s+name)?|name of patient|pt\.?\s*name)\s*[:\-]\s*"
        r"([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,3})",
        re.I,
    ),
    re.compile(
        r"\b(?:mr|mrs|ms|miss|dr)\.?\s+([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,3})"
    ),
)

FIELD_PATTERNS = {
    "condition": re.compile(r"(?:condition|complaint|impression)\s*[:\-]\s*(.+)", re.I),
    "diagnosis": re.compile(r"(?:diagnosis|diagnosed(?:\s+with)?)\s*[:\-]\s*(.+)", re.I),
    "treatment": re.compile(r"(?:treatment|plan|advised)\s*[:\-]\s*(.+)", re.I),
    "medications": re.compile(r"(?:medications?|drugs?|rx)\s*[:\-]\s*(.+)", re.I),
}


@dataclass
class ParsedRecord:
    is_medical_record: bool
    patient_name: str | None = None
    condition: str | None = None
    diagnosis: str | None = None
    treatment: str | None = None
    medications: str | None = None
    notes: str | None = None
    record_type: str = "note"
    alert_level: str = "none"
    hints: list[str] = field(default_factory=list)


def wants_to_file(question: str) -> bool:
    text = (question or "").lower()
    return any(phrase in text for phrase in FILE_INTENT)


def is_medical_record(text: str, question: str = "") -> bool:
    if wants_to_file(question):
        return True
    body = (text or "").lower()
    hits = [hint for hint in RECORD_HINTS if hint in body]
    return len(hits) >= 2


def extract_patient_name(text: str) -> str | None:
    blob = text or ""
    for pattern in NAME_PATTERNS:
        match = pattern.search(blob)
        if match:
            name = _clean_captured_name(match.group(1))
            if name:
                return name
    return None


def _clean_captured_name(raw: str) -> str | None:
    parts = []
    for part in re.sub(r"\s+", " ", raw).replace(":", " ").split():
        if part.lower() in NAME_STOP:
            break
        parts.append(part)
    name = " ".join(parts).strip(" .,-")
    return name if _plausible_name(name) else None


def parse_record(text: str, question: str = "") -> ParsedRecord:
    """Regex parse first; optionally refine with the eval model when a key is set."""
    body = (text or "").strip()
    hints = [hint for hint in RECORD_HINTS if hint in body.lower()]
    parsed = ParsedRecord(
        is_medical_record=is_medical_record(body, question),
        patient_name=extract_patient_name(body),
        notes=body[:1200] if body else None,
        hints=hints,
    )
    for field_name, pattern in FIELD_PATTERNS.items():
        match = pattern.search(body)
        if match:
            setattr(parsed, field_name, match.group(1).strip()[:240])

    if parsed.diagnosis:
        parsed.record_type = "diagnosis"
    elif any(word in body.lower() for word in ("lab", "hemoglobin", "creatinine", "investigation")):
        parsed.record_type = "lab_result"
    elif parsed.medications:
        parsed.record_type = "medication"

    lowered = body.lower()
    if any(word in lowered for word in ("critical", "emergency", "severe", "life-threatening")):
        parsed.alert_level = "high"
    elif any(word in lowered for word in ("abnormal", "elevated", "caution")):
        parsed.alert_level = "medium"

    return parsed


def looks_like_name_reply(text: str) -> bool:
    """A short follow-up that is likely naming the patient for a waiting file."""
    raw = (text or "").strip()
    if not raw:
        return False
    lowered = raw.lower()
    if any(
        phrase in lowered
        for phrase in ("this is for", "belongs to", "patient is", "add to", "for my", "name is")
    ):
        return True
    words = raw.split()
    return 1 <= len(words) <= 6 and not any(ch in raw for ch in "?!")


def _plausible_name(name: str) -> bool:
    parts = name.split()
    if not parts or len(parts) > 4:
        return False
    blocked = {"patient", "name", "date", "male", "female", "age", "report", "hospital"}
    return all(
        part.lower() not in blocked
        and part.replace("-", "").replace(".", "").isalpha()
        for part in parts
    )
