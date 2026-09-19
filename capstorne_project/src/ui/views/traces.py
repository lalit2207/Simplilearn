"""Admin view: agent memory, planning traces, and tool logs.

This is the capstone's "Memory and Logs Interface": memory traces, planning
breakdowns, interactive scenario testing, and tool usage logged with
success/failure across tasks.

Live traces come from this session's objects; historical traces are read back
from logs/agent_runs.jsonl so they survive a restart.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.agent.memory import long_term_patient_context
from src.llmops import load_runs, log_run
from src.llmops.run_log import iter_tool_calls
from src.tools.patient_tools import list_patients
from src.ui.components import dataframe, metric_row, page_header, render_trace
from src.ui.state import get_agent, get_memory, record_trace, session_traces

TEST_SCENARIOS = {
    "Multi-step: book + summarise (capstone sample)": (
        "My 70-year-old father has chronic kidney disease. I want to book a "
        "nephrologist for him. Also, can you summarize latest treatment methods?"
    ),
    "Single tool: resolve a relation": "Who is my father?",
    "History retrieval": "What is my father's most recent kidney test result?",
    "Search only": "Why does diabetes damage the kidneys?",
    "Safety refusal": (
        "My father's blood pressure is still high. Should I double his telmisartan dose?"
    ),
    "Unknown patient (should fail gracefully)": "Book a dermatologist for my uncle.",
    "Impossible specialty (should fail gracefully)": "Book a neurosurgeon for my father.",
}


def _live_traces_tab() -> None:
    traces = session_traces()
    if not traces:
        st.info(
            "No requests in this session yet. Use the Assistant page or the scenario "
            "runner below."
        )
        return

    labels = {
        f"{index}. [{trace.timestamp}] {trace.question[:70]}": index - 1
        for index, trace in enumerate(traces, start=1)
    }
    choice = st.selectbox("Request", list(reversed(labels)))
    render_trace(traces[labels[choice]], expanded=True)


def _memory_tab() -> None:
    memory = get_memory()

    metric_row(
        [
            ("Turns held", len(memory.turns)),
            ("Patient in focus", memory.last_patient_id or "-"),
            ("Thread", memory.thread_id.split("-")[-1]),
        ]
    )

    st.subheader("Short-term memory")
    st.caption(
        "The recent conversation, windowed to MEMORY_WINDOW turns. Cleared when the "
        "session resets."
    )
    if not memory.turns:
        st.info("No conversation turns recorded yet.")
    else:
        for index, turn in enumerate(reversed(memory.turns), start=1):
            with st.expander(
                f"{len(memory.turns) - index + 1}. {turn.question[:70]} "
                f"({turn.timestamp})"
            ):
                st.markdown(f"**Attendant:** {turn.question}")
                st.markdown(f"**Assistant:** {turn.answer}")
                st.caption(f"Patient id: {turn.patient_id or 'not identified'}")

    st.divider()
    st.subheader("Long-term memory")
    st.caption(
        "Retrieved from SQLite and FAISS by relevance to a query, not by recency. "
        "This works on a fresh session and survives a restart."
    )

    patients = list_patients()
    labels = {f"{p['full_name']} (id {p['id']})": p["id"] for p in patients}
    left, right = st.columns([1, 2])
    patient_choice = left.selectbox("Patient", list(labels))
    query = right.text_input("Query to retrieve against", value="kidney disease treatment")

    if query:
        context = long_term_patient_context(labels[patient_choice], query)
        st.code(context or "No context retrieved.", language="text")


def _scenario_tab() -> None:
    st.caption(
        "Run a scenario and inspect exactly how it was processed. Failure scenarios "
        "are included on purpose - graceful degradation is part of the design."
    )
    label = st.selectbox("Scenario", list(TEST_SCENARIOS))
    question = st.text_area("Request sent to the agent", value=TEST_SCENARIOS[label], height=90)

    left, right = st.columns(2)
    use_planner = left.toggle("LLM planner", value=True)
    plan_only = right.toggle("Plan only (no tool execution)", value=False)

    if st.button("Run scenario", type="primary", width="stretch"):
        if plan_only:
            from src.agent.memory import build_memory_section
            from src.agent.planner import create_plan

            memory = get_memory()
            with st.spinner("Planning..."):
                plan = create_plan(
                    question,
                    context=build_memory_section(memory, question, memory.last_patient_id),
                    use_llm=use_planner,
                )
            st.success(f"Plan produced by: {plan.source}")
            st.json(plan.as_dict())
        else:
            memory = get_memory()
            with st.spinner("Running the agent..."):
                trace = get_agent().run(question, use_planner=use_planner)
            record_trace(trace)
            log_run(trace, session_id=memory.thread_id)
            render_trace(trace, expanded=True)


def _history_tab() -> None:
    runs = load_runs()
    if not runs:
        st.info("No runs logged yet. Logs are written to logs/agent_runs.jsonl.")
        return

    st.caption(f"{len(runs)} runs logged. Newest first.")
    dataframe(
        [
            {
                "Time": run.get("timestamp"),
                "Question": (run.get("question") or "")[:70],
                "Success": run.get("succeeded"),
                "Plan": run.get("plan_source"),
                "Steps": run.get("planned_steps"),
                "Tools": len(run.get("tool_calls", [])),
                "Failures": run.get("tool_failures", 0),
                "Duration (s)": round(run.get("duration_ms", 0) / 1000, 1),
                "Eval case": run.get("eval_case", "-"),
            }
            for run in reversed(runs)
        ]
    )

    st.subheader("Tool usage log")
    st.caption("Every tool call across all logged runs, with success and failure.")
    calls = list(iter_tool_calls(runs))
    if calls:
        frame = pd.DataFrame(
            [
                {
                    "Time": call.get("timestamp"),
                    "Tool": call.get("name"),
                    "Module": call.get("module"),
                    "OK": call.get("ok"),
                    "Output": (call.get("output") or "")[:90],
                }
                for call in reversed(calls)
            ]
        )
        only_failures = st.toggle("Show failures only", value=False)
        if only_failures:
            frame = frame[~frame["OK"]]
        st.dataframe(frame, width="stretch", hide_index=True)


def render() -> None:
    page_header(
        "Agent Traces, Memory and Logs",
        "How each request was planned, which tools ran, and what the agent remembers.",
    )

    live, memory, scenarios, history = st.tabs(
        ["Session traces", "Memory inspector", "Scenario runner", "Logged history"]
    )
    with live:
        _live_traces_tab()
    with memory:
        _memory_tab()
    with scenarios:
        _scenario_tab()
    with history:
        _history_tab()
