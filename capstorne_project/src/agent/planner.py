"""Goal decomposition: turn one request into an ordered list of sub-goals.

Planning is a separate LLM call rather than something folded into the agent
loop, for two reasons the capstone cares about:

  * the plan is inspectable, so the Streamlit UI can show the breakdown, and
  * a wrong plan is visible before any tool mutates data.

If the model returns unusable JSON, `heuristic_plan` produces a keyword-based
plan instead. A degraded plan is far better than a crashed request.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from src.agent.llm import get_llm
from src.agent.prompts import PLANNER_PROMPT, tool_catalogue
from src.tools.registry import ALL_TOOLS, TOOLS_BY_NAME

BOOKING_WORDS = ("book", "appointment", "schedule", "slot", "consult", "visit", "see a")
HISTORY_WORDS = ("history", "summar", "past", "previous", "record", "chart", "diagnos")
SEARCH_WORDS = ("latest", "treatment", "what is", "symptom", "cause", "research", "guideline", "explain")
ADD_WORDS = ("add", "update", "record that", "note that", "log ", "attached", "file this")
CANCEL_WORDS = ("cancel", "reschedul")
PATIENT_WORDS = (
    "my ", "father", "mother", "dad", "mom", "wife", "husband", "son",
    "daughter", "brother", "sister", "patient", "for me",
)


@dataclass
class PlanStep:
    step: int
    sub_goal: str
    tool: str
    why: str = ""

    @property
    def tool_exists(self) -> bool:
        return self.tool == "none" or self.tool in TOOLS_BY_NAME


@dataclass
class Plan:
    goal: str
    steps: list[PlanStep]
    source: str = "llm"  # llm | heuristic | llm-repaired

    def as_dict(self) -> dict:
        return {"goal": self.goal, "source": self.source, "steps": [asdict(s) for s in self.steps]}

    def to_prompt_text(self) -> str:
        if not self.steps:
            return "No explicit plan; work directly from the request."
        lines = [f"Goal: {self.goal}"]
        for step in self.steps:
            suffix = "" if step.tool == "none" else f" [tool: {step.tool}]"
            lines.append(f"  {step.step}. {step.sub_goal}{suffix}")
        return "\n".join(lines)

    @property
    def tools_planned(self) -> list[str]:
        return [s.tool for s in self.steps if s.tool != "none"]


def _extract_json(text: str) -> dict | None:
    """Pull a JSON object out of a model response.

    Models wrap JSON in code fences or add a sentence before it even when told
    not to, so we locate the outermost braces rather than trusting the shape.
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def heuristic_plan(question: str) -> Plan:
    """Keyword fallback plan, used when the LLM is unavailable or returns junk."""
    text = question.lower()
    steps: list[PlanStep] = []

    def add(sub_goal: str, tool: str, why: str) -> None:
        steps.append(PlanStep(len(steps) + 1, sub_goal, tool, why))

    needs_patient = any(word in text for word in PATIENT_WORDS)
    person_task = any(
        word in text for word in (*HISTORY_WORDS, *ADD_WORDS, *BOOKING_WORDS, *CANCEL_WORDS)
    )
    if needs_patient:
        add("Identify which family member the request refers to", "resolve_patient",
            "Name or relation must become a numeric patient_id first")
    elif person_task:
        add("List registered family members so the attendant can choose",
            "list_family_members",
            "The request is about a person but no name or relation was given")
        add("Ask which family member this is for", "none",
            "Do not book or open a chart until they name someone")

    if any(word in text for word in HISTORY_WORDS):
        add("Retrieve and review the patient's medical history", "get_patient_history",
            "The request asks about past records")

    if any(word in text for word in ADD_WORDS):
        add("Add the new information to the patient's chart", "add_patient_record",
            "The request asks to record something new")

    if any(word in text for word in CANCEL_WORDS):
        add("Find the existing appointment", "list_appointments",
            "Cancelling requires the appointment_id")
        add("Cancel the appointment and release the slot", "cancel_appointment",
            "The request asks to cancel or reschedule")

    if any(word in text for word in BOOKING_WORDS):
        add("Find open slots with a suitable specialist", "find_available_slots",
            "Booking needs a valid slot_id")
        add("Book the chosen slot for the patient", "book_appointment",
            "The request asks for an appointment")

    if any(word in text for word in SEARCH_WORDS):
        add("Look up current information from MedlinePlus and WHO",
            "search_medical_information", "The request asks for medical information")

    if not steps:
        add("Answer the attendant's question directly", "none",
            "No tool-specific intent detected")

    return Plan(goal=question.strip(), steps=steps, source="heuristic")


def doctor_heuristic_plan(question: str) -> Plan:
    """Fallback plan for the doctor assistant (patient lookup, not booking)."""
    text = question.lower()
    steps: list[PlanStep] = []

    def add(sub_goal: str, tool: str, why: str) -> None:
        steps.append(PlanStep(len(steps) + 1, sub_goal, tool, why))

    schedule_words = ("my list", "today", "appointment", "who is", "who are", "schedule")
    person_hint = any(
        word in text
        for word in ("ramesh", "sunita", "lalit", "chauhan", "patient", "father", "mother")
    ) or any(word in text for word in HISTORY_WORDS)

    if any(word in text for word in schedule_words) and not person_hint:
        add("List the doctor's appointments", "list_doctor_appointments",
            "The request is about the clinic list")
    elif person_hint or any(word in text for word in (*HISTORY_WORDS, *ADD_WORDS, "allerg")):
        add("Find the named patient", "search_patients",
            "A chart lookup needs a numeric patient_id")
        if any(word in text for word in HISTORY_WORDS) or "allerg" in text:
            add("Retrieve the patient's medical history", "get_patient_history",
                "The request asks about the chart")
        if any(word in text for word in ADD_WORDS):
            add("Add a note to the patient's chart", "add_patient_record",
                "The request asks to record something")

    if any(word in text for word in SEARCH_WORDS):
        add("Look up current information from MedlinePlus and WHO",
            "search_medical_information", "The request asks for medical information")

    if not steps:
        add("Answer the doctor's question directly", "none",
            "No tool-specific intent detected")

    return Plan(goal=question.strip(), steps=steps, source="heuristic")


def guest_heuristic_plan(question: str) -> Plan:
    """Guest visitors may only look up doctors, open slots, and medical info."""
    text = question.lower()
    steps: list[PlanStep] = []

    def add(sub_goal: str, tool: str, why: str) -> None:
        steps.append(PlanStep(len(steps) + 1, sub_goal, tool, why))

    gated = any(
        word in text
        for word in (
            *ADD_WORDS,
            *CANCEL_WORDS,
            "book",
            "history",
            "allerg",
            "my father",
            "my mother",
            "chart",
            "record that",
        )
    )
    if gated:
        add(
            "Ask the visitor to sign in as a registered user",
            "none",
            "Booking and patient charts are not available to guests",
        )

    if any(word in text for word in ("specialist", "doctor", "nephrolog", "cardiolog", "who can")):
        add("List matching doctors", "find_doctors", "The request asks who is available")
    if any(word in text for word in ("slot", "available", "next week", "appointment time", "when")):
        add("List open appointment times", "find_available_slots",
            "Guests may browse times but cannot book")
    if any(word in text for word in SEARCH_WORDS):
        add("Look up current information from MedlinePlus and WHO",
            "search_medical_information", "The request asks for medical information")

    if not steps:
        add("Answer the visitor's question directly", "none",
            "No tool-specific intent detected")

    return Plan(goal=question.strip(), steps=steps, source="heuristic")


def _validate(plan: Plan, fallback=heuristic_plan, allowed: set[str] | None = None) -> Plan:
    """Drop steps naming tools that do not exist, renumbering what remains."""
    def ok(step: PlanStep) -> bool:
        if step.tool == "none":
            return True
        if allowed is not None:
            return step.tool in allowed
        return step.tool_exists

    kept = [step for step in plan.steps if ok(step)]
    for index, step in enumerate(kept, start=1):
        step.step = index
    if not kept:
        return fallback(plan.goal)
    plan.steps = kept
    return plan


def create_plan(
    question: str,
    context: str = "",
    use_llm: bool = True,
    tools=None,
    role: str = "attendant",
) -> Plan:
    """Decompose `question` into sub-goals.

    Falls back to the heuristic plan on a missing key, a network error, or
    unparseable output, so planning never becomes a hard failure point.
    """
    catalogue = list(tools or ALL_TOOLS)
    if role == "doctor":
        fallback = doctor_heuristic_plan
    elif role == "guest":
        fallback = guest_heuristic_plan
    else:
        fallback = heuristic_plan
    allowed = {getattr(tool, "name", "") for tool in catalogue}

    if not use_llm:
        return fallback(question)

    prompt = PLANNER_PROMPT.format(
        tool_catalogue=tool_catalogue(catalogue),
        context=context or "None yet.",
        question=question,
    )

    try:
        response = get_llm(temperature=0.0).invoke(prompt)
        payload = _extract_json(getattr(response, "content", "") or "")
    except Exception:
        return fallback(question)

    if not payload or not isinstance(payload.get("steps"), list):
        return fallback(question)

    steps: list[PlanStep] = []
    for index, raw in enumerate(payload["steps"], start=1):
        if not isinstance(raw, dict):
            continue
        steps.append(
            PlanStep(
                step=int(raw.get("step", index) or index),
                sub_goal=str(raw.get("sub_goal", "")).strip() or "Unnamed sub-goal",
                tool=str(raw.get("tool", "none")).strip() or "none",
                why=str(raw.get("why", "")).strip(),
            )
        )

    if not steps:
        return fallback(question)

    return _validate(
        Plan(goal=str(payload.get("goal", question)).strip(), steps=steps),
        fallback=fallback,
        allowed=allowed,
    )
