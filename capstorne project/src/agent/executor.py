"""The agent execution flow: plan, then act, then report.

This implements the capstone's sample workflow end to end:

    1. identify patient and context   -> resolve_patient + memory lookup
    2. retrieve medical history       -> get_patient_history
    3. query doctor calendar and book -> find_available_slots + book_appointment
    4. search and summarise via RAG   -> search_medical_information

Every tool call is captured in an `AgentTrace` so Step 6 can score the run and
Step 7 can display the reasoning without re-running anything.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.agents import create_agent

from src.agent.llm import MissingAPIKeyError, get_llm
from src.agent.memory import SessionMemory, build_memory_section, get_checkpointer
from src.agent.planner import Plan, create_plan
from src.agent.prompts import AGENT_PROMPT, DOCTOR_AGENT_PROMPT, GUEST_AGENT_PROMPT, SAFETY_RULES
from src.tools.registry import ALL_TOOLS, DOCTOR_TOOLS, GUEST_TOOLS

# Cap the tool-calling loop. Without this a confused model can ping-pong
# between two tools until the request times out.
MAX_ITERATIONS = 12


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict
    output: str
    duration_ms: int = 0
    ok: bool = True

    @property
    def preview(self) -> str:
        return self.output if len(self.output) <= 400 else self.output[:400] + "..."


@dataclass
class AgentTrace:
    """Everything that happened during one request."""

    question: str
    plan: Plan | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    answer: str = ""
    patient_id: int | None = None
    error: str | None = None
    duration_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def succeeded(self) -> bool:
        return self.error is None and bool(self.answer)

    @property
    def tools_used(self) -> list[str]:
        return [call.name for call in self.tool_calls]

    @property
    def failed_calls(self) -> list[ToolCallRecord]:
        return [call for call in self.tool_calls if not call.ok]

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "question": self.question,
            "plan": self.plan.as_dict() if self.plan else None,
            "tool_calls": [
                {
                    "name": c.name,
                    "arguments": c.arguments,
                    "output": c.output,
                    "duration_ms": c.duration_ms,
                    "ok": c.ok,
                }
                for c in self.tool_calls
            ],
            "answer": self.answer,
            "patient_id": self.patient_id,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "succeeded": self.succeeded,
        }


# Tool outputs are prose, so failure has to be detected from wording. The
# wrappers in registry.py were written to start failures with these phrases.
_FAILURE_PATTERNS = (
    "failed",
    "could not identify",
    "no results found",
    "no available slots",
    "no doctors found",
    "no appointments found",
    "no patient",
)


def _looks_like_failure(output: str) -> bool:
    lowered = output.lower()
    return any(pattern in lowered for pattern in _FAILURE_PATTERNS)


def _extract_patient_id(trace: AgentTrace) -> int | None:
    """Recover the patient id the agent settled on, for memory and logging."""
    for call in trace.tool_calls:
        if call.name in ("resolve_patient", "search_patients") and call.ok:
            match = re.search(r"\(id (\d+)\)", call.output)
            if match:
                return int(match.group(1))
        for key in ("patient_id",):
            if isinstance(call.arguments.get(key), int):
                return call.arguments[key]
    return None


class HealthcareAgent:
    """Planner + tool-calling agent over the patient, scheduling, and RAG tools."""

    def __init__(self, memory: SessionMemory, tools=None, prompt_template: str | None = None):
        self.memory = memory
        self.tools = list(tools or ALL_TOOLS)
        self.prompt_template = prompt_template

    def _build_graph(self, system_prompt: str):
        """Compile the graph for this request.

        `system_prompt` is a constructor argument, not runtime config, so the
        graph is rebuilt each turn - the plan and memory block differ every
        time. Compilation is local and cheap; conversation continuity comes
        from the shared checkpointer keyed by thread_id, not from reusing the
        graph object.
        """
        return create_agent(
            model=get_llm(),
            tools=self.tools,
            system_prompt=system_prompt,
            checkpointer=get_checkpointer(),
        )

    def run(self, question: str, use_planner: bool = True) -> AgentTrace:
        """Plan the request, execute it, and return the full trace."""
        started = time.perf_counter()
        trace = AgentTrace(question=question)

        try:
            memory_section = build_memory_section(
                self.memory, question, self.memory.last_patient_id
            )
            trace.plan = create_plan(
                question,
                context=memory_section,
                use_llm=use_planner,
                tools=self.tools,
                role=self.memory.role,
            )

            if self.memory.role == "doctor":
                template = self.prompt_template or DOCTOR_AGENT_PROMPT
                system_prompt = template.format(
                    safety_rules=SAFETY_RULES,
                    doctor_name=self.memory.doctor_name or self.memory.attendant_name,
                    doctor_id=self.memory.doctor_id or 0,
                    today=date.today().isoformat(),
                    memory_section=memory_section,
                    plan=trace.plan.to_prompt_text(),
                )
            elif self.memory.role == "guest":
                template = self.prompt_template or GUEST_AGENT_PROMPT
                system_prompt = template.format(
                    safety_rules=SAFETY_RULES,
                    attendant_name=self.memory.attendant_name,
                    today=date.today().isoformat(),
                    memory_section=memory_section,
                    plan=trace.plan.to_prompt_text(),
                )
            else:
                template = self.prompt_template or AGENT_PROMPT
                system_prompt = template.format(
                    safety_rules=SAFETY_RULES,
                    attendant_name=self.memory.attendant_name,
                    attendant_id=self.memory.attendant_id,
                    today=date.today().isoformat(),
                    memory_section=memory_section,
                    plan=trace.plan.to_prompt_text(),
                )

            graph = self._build_graph(system_prompt)
            result = graph.invoke(
                {"messages": [HumanMessage(content=question)]},
                config={
                    "configurable": {"thread_id": self.memory.thread_id},
                    "recursion_limit": MAX_ITERATIONS * 2,
                },
            )
            self._collect(result, trace)

        except MissingAPIKeyError as exc:
            trace.error = str(exc)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
            trace.error = f"{type(exc).__name__}: {exc}"

        trace.duration_ms = int((time.perf_counter() - started) * 1000)
        trace.patient_id = _extract_patient_id(trace)
        if trace.succeeded:
            self.memory.record(question, trace.answer, trace.patient_id)
        return trace

    def _collect(self, result: dict[str, Any], trace: AgentTrace) -> None:
        """Turn the graph's message list into tool records plus a final answer.

        Tool calls and their results arrive as separate messages, so requested
        arguments are matched to outputs by tool_call_id.
        """
        messages = result.get("messages", [])
        requested: dict[str, tuple[str, dict]] = {}

        for message in messages:
            if isinstance(message, AIMessage):
                for call in message.tool_calls or []:
                    requested[call["id"]] = (call["name"], call.get("args", {}) or {})
            elif isinstance(message, ToolMessage):
                name, arguments = requested.get(
                    message.tool_call_id, (message.name or "unknown", {})
                )
                output = str(message.content)
                trace.tool_calls.append(
                    ToolCallRecord(
                        name=name,
                        arguments=arguments,
                        output=output,
                        ok=not _looks_like_failure(output),
                    )
                )

        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                trace.answer = str(message.content).strip()
                break

        if not trace.answer:
            trace.error = "The agent stopped without producing a final answer."


def build_session(
    attendant_id: int = 1,
    attendant_name: str = "Lalit Chauhan",
    thread_id: str | None = None,
) -> SessionMemory:
    return SessionMemory(
        attendant_id=attendant_id,
        attendant_name=attendant_name,
        thread_id=thread_id or f"session-{attendant_id}-{int(time.time())}",
    )


def build_doctor_session(
    doctor_id: int = 1,
    doctor_name: str = "Dr. Anil Mehta",
    thread_id: str | None = None,
) -> SessionMemory:
    return SessionMemory(
        attendant_id=0,
        attendant_name=doctor_name,
        thread_id=thread_id or f"doctor-{doctor_id}-{int(time.time())}",
        role="doctor",
        doctor_id=doctor_id,
        doctor_name=doctor_name,
    )


def build_doctor_agent(memory: SessionMemory) -> HealthcareAgent:
    return HealthcareAgent(memory, tools=DOCTOR_TOOLS, prompt_template=DOCTOR_AGENT_PROMPT)


def build_guest_session(thread_id: str | None = None) -> SessionMemory:
    return SessionMemory(
        attendant_id=0,
        attendant_name="Guest",
        thread_id=thread_id or f"guest-{int(time.time())}",
        role="guest",
    )


def build_guest_agent(memory: SessionMemory) -> HealthcareAgent:
    return HealthcareAgent(memory, tools=GUEST_TOOLS, prompt_template=GUEST_AGENT_PROMPT)
