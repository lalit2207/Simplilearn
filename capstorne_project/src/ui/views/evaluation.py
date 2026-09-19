"""Admin view: evaluation metrics and per-module performance.

Covers "evaluation metrics for model responses and tool success". Charts are
driven entirely by the JSONL logs, so this page works without re-running the
agent - and shows real numbers rather than placeholders.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import settings
from src.llmops import (
    booking_metrics,
    categories,
    evaluation_metrics,
    load_evaluations,
    module_metrics,
    overall_metrics,
    planning_metrics,
    tool_metrics,
)
from src.llmops.dataset import EVAL_CASES
from src.llmops.evaluation import evaluate_cases, summarise
from src.ui.components import dataframe, llm_warning, metric_row, page_header


def _performance_tab() -> None:
    overall = overall_metrics()
    if overall["runs"] == 0:
        st.info(
            "No runs logged yet. Use the Assistant page or run "
            "`python scripts/run_eval.py` to generate data."
        )
        return

    metric_row(
        [
            ("Runs", overall["runs"]),
            ("Run success", f"{overall['success_rate']:.0%}"),
            ("Tool success", f"{overall['tool_success_rate']:.0%}"),
            ("Avg duration", f"{overall['avg_duration_ms'] / 1000:.1f}s"),
            ("Avg tools/run", overall["avg_tools_per_run"]),
        ]
    )

    booking = booking_metrics()
    st.subheader("Booking success rate")
    st.caption(
        "Counts book_appointment calls only. A slot search is not a booking attempt."
    )
    metric_row(
        [
            ("Attempts", booking["attempts"]),
            ("Confirmed", booking["confirmed"]),
            ("Failed", booking["failed"]),
            ("Success rate", f"{booking['success_rate']:.0%}"),
        ]
    )

    st.subheader("Success rate per module")
    modules = module_metrics()
    if modules:
        frame = pd.DataFrame(modules).set_index("module")
        st.bar_chart(frame["success_rate"], height=260)
        dataframe(modules)

    st.subheader("Per-tool breakdown")
    dataframe(tool_metrics())

    st.subheader("Planning quality")
    planning = planning_metrics()
    st.caption(
        "Precision: how many planned tools were used. Recall: how many used tools were "
        "planned. Low recall means the agent improvised beyond its plan."
    )
    metric_row(
        [
            ("Runs with a plan", planning["runs_with_plan"]),
            ("LLM plans", planning["llm_plans"]),
            ("Heuristic plans", planning["heuristic_plans"]),
            ("Precision", f"{planning['avg_precision']:.0%}"),
            ("Recall", f"{planning['avg_recall']:.0%}"),
        ]
    )


def _quality_tab() -> None:
    summary = evaluation_metrics()
    if summary["evaluated"] == 0:
        st.info("No evaluations recorded yet. Run the evaluation from the next tab.")
        return

    metric_row(
        [
            ("Cases graded", summary["evaluated"]),
            ("Correct", summary["correct"]),
            ("Accuracy", f"{summary['accuracy']:.0%}"),
        ]
    )

    st.subheader("Accuracy per category")
    by_category = summary["by_category"]
    if by_category:
        frame = pd.DataFrame(by_category).set_index("category")
        st.bar_chart(frame["accuracy"], height=260)
        dataframe(by_category)

    st.subheader("Graded answers")
    records = load_evaluations()
    for record in reversed(records[-30:]):
        verdict = "PASS" if record.get("correct") else "FAIL"
        icon = ":material/check_circle:" if record.get("correct") else ":material/cancel:"
        with st.expander(
            f"[{verdict}] {record.get('case_id')} - {record.get('category')}", icon=icon
        ):
            st.markdown(f"**Question:** {record.get('question')}")
            st.markdown(f"**Reference:** {record.get('reference')}")
            st.markdown(f"**Agent answer:** {record.get('answer') or '(none)'}")
            metric_row(
                [
                    ("QAEvalChain", record.get("grade", "UNKNOWN")),
                    ("Keyword coverage", record.get("keyword_coverage", 0)),
                    ("Tool coverage", record.get("tool_coverage", 0)),
                ]
            )
            if record.get("missing_keywords"):
                st.warning(f"Missing keywords: {', '.join(record['missing_keywords'])}")
            if record.get("missing_tools"):
                st.warning(f"Tools never called: {', '.join(record['missing_tools'])}")
            if record.get("error"):
                st.error(record["error"])


def _runner_tab() -> None:
    st.caption(
        "Each case runs the full agent, then QAEvalChain grades the answer against a "
        "reference. Two deterministic checks run alongside it: keyword coverage catches "
        "omitted facts, tool coverage catches answers produced without consulting data."
    )
    llm_ready = llm_warning()

    st.markdown(
        f"**{len(EVAL_CASES)} cases** across: {', '.join(categories())}  \n"
        f"Agent model: `{settings.groq_model}` · Judge model: `{settings.groq_eval_model}`"
        "  \nSet GROQ_API_KEY in .env or Streamlit secrets if the button is disabled."
    )

    chosen = st.multiselect(
        "Categories to evaluate (empty means all)", categories(), default=[]
    )
    use_planner = st.toggle("LLM planner", value=True)

    selected = [case for case in EVAL_CASES if not chosen or case.category in chosen]
    st.caption(f"{len(selected)} case(s) selected. Expect roughly 3-8 seconds each.")

    if st.button(
        "Run evaluation", type="primary", disabled=not llm_ready, width="stretch"
    ):
        progress = st.progress(0.0, text="Starting...")
        results = []
        for index, case in enumerate(selected, start=1):
            progress.progress(
                (index - 1) / len(selected), text=f"Running {case.id} ({case.category})..."
            )
            results.extend(evaluate_cases([case], use_planner=use_planner))
        progress.progress(1.0, text="Grading complete.")

        st.success("Evaluation finished.")
        summary = summarise(results)
        metric_row(
            [
                ("Cases", summary["cases"]),
                ("Correct", summary["correct"]),
                ("Accuracy", f"{summary['accuracy']:.0%}"),
                ("Keyword coverage", f"{summary['avg_keyword_coverage']:.0%}"),
                ("Tool coverage", f"{summary['avg_tool_coverage']:.0%}"),
            ]
        )
        dataframe(
            [
                {
                    "Case": r.case_id,
                    "Category": r.category,
                    "Grade": r.grade,
                    "Keywords": r.keyword_coverage,
                    "Tools": r.tool_coverage,
                    "Duration (s)": round(r.duration_ms / 1000, 1),
                    "Error": r.error or "-",
                }
                for r in results
            ]
        )


def render() -> None:
    page_header(
        "Evaluation and Metrics",
        "Model response quality and tool success, per module.",
    )
    performance, quality, runner = st.tabs(
        ["Tool and module performance", "Response quality", "Run evaluation"]
    )
    with performance:
        _performance_tab()
    with quality:
        _quality_tab()
    with runner:
        _runner_tab()
