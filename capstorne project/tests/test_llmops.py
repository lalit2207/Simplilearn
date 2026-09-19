"""Logging, metrics, and grading tests - all offline.

Synthetic traces stand in for real agent runs so the metrics maths can be
checked against known-correct numbers. Logs are redirected to a temporary
directory so a test run never pollutes the real logs.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.executor import AgentTrace, ToolCallRecord  # noqa: E402
from src.agent.planner import heuristic_plan  # noqa: E402
from src.llmops import evaluation, metrics, run_log  # noqa: E402


@pytest.fixture
def temp_logs(tmp_path, monkeypatch):
    """Point the log directory at a temp folder for the duration of a test.

    `Settings` is a frozen dataclass, so its fields cannot be assigned. We
    swap in a modified copy instead - which is exactly the discipline the
    frozen config is there to enforce.
    """
    monkeypatch.setattr(
        run_log, "settings", replace(run_log.settings, log_dir=tmp_path)
    )
    return tmp_path


def _trace(question, tools, answer="Done.", error=None, duration=1000):
    """Build a synthetic trace. `tools` is a list of (name, ok) pairs."""
    trace = AgentTrace(question=question, answer=answer, error=error, duration_ms=duration)
    trace.plan = heuristic_plan(question)
    trace.tool_calls = [
        ToolCallRecord(name=name, arguments={}, output="ok" if ok else "Booking failed", ok=ok)
        for name, ok in tools
    ]
    return trace


def test_log_run_writes_readable_jsonl(temp_logs):
    trace = _trace("book a nephrologist for my father", [("resolve_patient", True)])
    run_log.log_run(trace, session_id="s1")

    runs = run_log.load_runs()
    assert len(runs) == 1
    assert runs[0]["question"] == "book a nephrologist for my father"
    assert runs[0]["succeeded"] is True
    assert runs[0]["tool_calls"][0]["module"] == "patient_identification"


def test_corrupt_log_line_is_skipped(temp_logs):
    run_log.log_run(_trace("q1", [("resolve_patient", True)]))
    (temp_logs / run_log.RUN_LOG_NAME).open("a", encoding="utf-8").write("{not json\n")
    run_log.log_run(_trace("q2", [("resolve_patient", True)]))

    # The bad line is dropped, the two good records survive.
    assert len(run_log.load_runs()) == 2


def test_module_mapping_groups_booking_tools_together():
    assert run_log.module_for("find_available_slots") == "appointment_booking"
    assert run_log.module_for("book_appointment") == "appointment_booking"
    assert run_log.module_for("search_medical_information") == "medical_search"
    assert run_log.module_for("nonexistent_tool") == "other"


def test_overall_and_module_metrics(temp_logs):
    run_log.log_run(_trace("book for my father", [("resolve_patient", True), ("book_appointment", True)]))
    run_log.log_run(_trace("book for my father", [("resolve_patient", True), ("book_appointment", False)]))

    overall = metrics.overall_metrics()
    assert overall["runs"] == 2
    assert overall["success_rate"] == 1.0          # both produced answers
    assert overall["tool_success_rate"] == 0.75    # 3 of 4 calls ok
    assert overall["runs_with_failures"] == 1

    by_module = {row["module"]: row for row in metrics.module_metrics()}
    assert by_module["patient_identification"]["success_rate"] == 1.0
    assert by_module["appointment_booking"]["success_rate"] == 0.5
    # Worst module is listed first so failures surface on the dashboard.
    assert metrics.module_metrics()[0]["module"] == "appointment_booking"


def test_booking_metrics_ignore_slot_searches(temp_logs):
    run_log.log_run(
        _trace(
            "book appointment",
            [("find_available_slots", True), ("book_appointment", True), ("book_appointment", False)],
        )
    )
    booking = metrics.booking_metrics()

    assert booking["attempts"] == 2      # slot search is not a booking attempt
    assert booking["confirmed"] == 1
    assert booking["success_rate"] == 0.5


def test_metrics_on_empty_logs_do_not_divide_by_zero(temp_logs):
    assert metrics.overall_metrics()["runs"] == 0
    assert metrics.overall_metrics()["success_rate"] == 0.0
    assert metrics.booking_metrics()["success_rate"] == 0.0
    assert metrics.evaluation_metrics()["accuracy"] == 0.0
    assert metrics.module_metrics() == []


def test_planning_precision_and_recall(temp_logs):
    # Heuristic plan for this question plans 5 tools; the agent used 2 of them.
    trace = _trace(
        "book a nephrologist for my father and summarize treatments",
        [("resolve_patient", True), ("book_appointment", True)],
    )
    run_log.log_run(trace)

    planning = metrics.planning_metrics()
    assert planning["runs_with_plan"] == 1
    assert planning["heuristic_plans"] == 1
    assert 0.0 < planning["avg_precision"] < 1.0   # not every planned tool was used
    assert planning["avg_recall"] == 1.0           # every used tool was planned


def test_keyword_coverage_detects_omitted_facts():
    score, missing = evaluation.keyword_coverage(
        "Your father Ramesh Chauhan is on file.", ["Ramesh", "70"]
    )
    assert score == 0.5
    assert missing == ["70"]

    score, missing = evaluation.keyword_coverage("anything", [])
    assert score == 1.0 and missing == []


def test_tool_coverage_detects_skipped_tools():
    score, missing = evaluation.tool_coverage(
        ["resolve_patient"], ["find_available_slots", "book_appointment"]
    )
    assert score == 0.0
    assert missing == ["find_available_slots", "book_appointment"]


@pytest.mark.parametrize(
    ("raw", "expected_grade", "expected_correct"),
    [
        ({"results": "CORRECT"}, "CORRECT", True),
        ({"results": "INCORRECT"}, "INCORRECT", False),
        ({"results": "GRADE: correct"}, "CORRECT", True),
        # "CORRECT" is a substring of "INCORRECT", so ordering matters here.
        ({"results": "GRADE: INCORRECT"}, "INCORRECT", False),
        ({"results": "unclear"}, "UNKNOWN", False),
        (None, "UNKNOWN", False),
    ],
)
def test_grade_parsing_handles_judge_phrasing(raw, expected_grade, expected_correct):
    grade, correct = evaluation._parse_grade(raw)
    assert (grade, correct) == (expected_grade, expected_correct)


def test_evaluation_metrics_group_by_category(temp_logs):
    run_log.log_evaluation({"case_id": "a", "category": "medical_search", "correct": True})
    run_log.log_evaluation({"case_id": "b", "category": "medical_search", "correct": False})
    run_log.log_evaluation({"case_id": "c", "category": "safety", "correct": True})

    summary = metrics.evaluation_metrics()
    assert summary["evaluated"] == 3
    assert summary["accuracy"] == pytest.approx(0.6667, abs=1e-4)

    by_category = {row["category"]: row["accuracy"] for row in summary["by_category"]}
    assert by_category["medical_search"] == 0.5
    assert by_category["safety"] == 1.0


def test_dataset_cases_are_well_formed():
    from src.llmops.dataset import EVAL_CASES
    from src.tools.registry import TOOLS_BY_NAME

    assert len({case.id for case in EVAL_CASES}) == len(EVAL_CASES)
    for case in EVAL_CASES:
        assert case.question and case.reference
        for tool in case.needs_tools:
            assert tool in TOOLS_BY_NAME, f"{case.id} requires unknown tool {tool}"
