"""Agent loop tests that run without a Groq API key.

A scripted chat model stands in for Groq so the plumbing - system prompt
assembly, the tool-calling loop, trace collection, and memory writes - can be
verified deterministically and offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.agent.executor as executor_module  # noqa: E402
from src.agent.executor import HealthcareAgent, build_session  # noqa: E402
from src.agent.planner import guest_heuristic_plan, heuristic_plan  # noqa: E402
from src.tools.registry import GUEST_TOOLS  # noqa: E402

# Captured out-of-band because pydantic models reject stray class attributes.
CAPTURED: dict[str, object] = {}


class ScriptedModel(BaseChatModel):
    """Requests one tool call, then produces a final answer."""

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        CAPTURED["tool_names"] = [getattr(t, "name", str(t)) for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # noqa: ANN001, ANN003
        for message in messages:
            if isinstance(message, SystemMessage):
                CAPTURED["system_prompt"] = message.content

        turn = int(CAPTURED.get("turn", 0)) + 1
        CAPTURED["turn"] = turn

        if turn == 1:
            reply = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "resolve_patient",
                        "args": {"reference": "my father", "attendant_id": 1},
                        "id": "call_1",
                    }
                ],
            )
        else:
            reply = AIMessage(content="Your father Ramesh Chauhan is on file.")
        return ChatResult(generations=[ChatGeneration(message=reply)])


@pytest.fixture
def scripted_agent(monkeypatch):
    CAPTURED.clear()
    monkeypatch.setattr(executor_module, "get_llm", lambda *a, **k: ScriptedModel())
    session = build_session(attendant_id=1, attendant_name="Lalit Chauhan")
    return HealthcareAgent(session), session


def test_agent_completes_and_records_trace(scripted_agent):
    agent, _ = scripted_agent
    trace = agent.run("book a nephrologist for my father", use_planner=False)

    assert trace.succeeded, trace.error
    assert trace.tools_used == ["resolve_patient"]
    assert trace.tool_calls[0].ok
    assert "Ramesh Chauhan" in trace.tool_calls[0].output
    assert trace.answer
    assert trace.duration_ms >= 0


def test_system_prompt_carries_rules_plan_and_context(scripted_agent):
    agent, _ = scripted_agent
    agent.run("book a nephrologist for my father", use_planner=False)

    system_prompt = CAPTURED["system_prompt"]
    assert "Safety rules" in system_prompt
    assert "attendant_id=1" in system_prompt
    assert "Goal:" in system_prompt
    assert "find_available_slots" in system_prompt


def test_all_tools_are_bound(scripted_agent):
    agent, _ = scripted_agent
    agent.run("book a nephrologist for my father", use_planner=False)

    assert "resolve_patient" in CAPTURED["tool_names"]
    assert "search_medical_information" in CAPTURED["tool_names"]
    assert len(CAPTURED["tool_names"]) == 11
    assert "list_family_members" in CAPTURED["tool_names"]
    assert "register_patient" in CAPTURED["tool_names"]


def test_patient_id_extracted_and_stored_in_memory(scripted_agent):
    agent, session = scripted_agent
    trace = agent.run("book a nephrologist for my father", use_planner=False)

    assert trace.patient_id == 1
    assert len(session.turns) == 1
    assert session.last_patient_id == 1


def test_missing_api_key_is_reported_not_raised(monkeypatch):
    from src.agent.llm import MissingAPIKeyError

    def refuse(*args, **kwargs):
        raise MissingAPIKeyError("GROQ_API_KEY is missing")

    monkeypatch.setattr(executor_module, "get_llm", refuse)
    trace = HealthcareAgent(build_session()).run("book something", use_planner=False)

    assert not trace.succeeded
    assert "GROQ_API_KEY" in trace.error
    # Planning still happened, so the UI can show the breakdown even when the
    # model is unavailable.
    assert trace.plan is not None and trace.plan.steps


def test_heuristic_plan_matches_capstone_sample_workflow():
    plan = heuristic_plan(
        "My 70-year-old father has chronic kidney disease. I want to book a "
        "nephrologist for him. Also, can you summarize latest treatment methods?"
    )
    tools = plan.tools_planned

    assert tools.index("resolve_patient") == 0
    assert tools.index("find_available_slots") < tools.index("book_appointment")
    assert "get_patient_history" in tools
    assert "search_medical_information" in tools


def test_guest_tools_cannot_book_or_open_charts():
    names = {tool.name for tool in GUEST_TOOLS}
    assert names == {"find_doctors", "find_available_slots", "search_medical_information"}
    plan = guest_heuristic_plan(
        "Book a nephrologist for my father and summarise his medical history."
    )
    assert "book_appointment" not in plan.tools_planned
    assert "get_patient_history" not in plan.tools_planned
    assert any(step.tool == "none" for step in plan.steps)
