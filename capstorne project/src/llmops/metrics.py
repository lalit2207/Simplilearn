"""Aggregate metrics computed from the run and evaluation logs.

The capstone asks for "success rate of bookings, response precision" per
module. Everything here is derived from the JSONL logs, so metrics never
require re-running the agent.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from src.llmops.run_log import iter_tool_calls, load_evaluations, load_runs


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(numerator / denominator, 4)


def overall_metrics(runs: list[dict] | None = None) -> dict:
    """Headline numbers for the dashboard."""
    runs = load_runs() if runs is None else runs
    if not runs:
        return {
            "runs": 0,
            "success_rate": 0.0,
            "avg_duration_ms": 0,
            "avg_tools_per_run": 0.0,
            "tool_success_rate": 0.0,
            "runs_with_failures": 0,
        }

    successes = sum(1 for run in runs if run.get("succeeded"))
    calls = list(iter_tool_calls(runs))
    ok_calls = sum(1 for call in calls if call.get("ok"))

    return {
        "runs": len(runs),
        "success_rate": _rate(successes, len(runs)),
        "avg_duration_ms": int(mean(run.get("duration_ms", 0) for run in runs)),
        "avg_tools_per_run": round(len(calls) / len(runs), 2),
        "tool_success_rate": _rate(ok_calls, len(calls)),
        "runs_with_failures": sum(1 for run in runs if run.get("tool_failures", 0) > 0),
    }


def module_metrics(runs: list[dict] | None = None) -> list[dict]:
    """Per-module call counts and success rates, worst first.

    Sorting by success rate ascending puts whatever is broken at the top of
    the dashboard instead of burying it.
    """
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "ok": 0})
    for call in iter_tool_calls(runs):
        bucket = grouped[call.get("module", "other")]
        bucket["calls"] += 1
        bucket["ok"] += 1 if call.get("ok") else 0

    rows = [
        {
            "module": module,
            "calls": stats["calls"],
            "successes": stats["ok"],
            "failures": stats["calls"] - stats["ok"],
            "success_rate": _rate(stats["ok"], stats["calls"]),
        }
        for module, stats in grouped.items()
    ]
    return sorted(rows, key=lambda row: (row["success_rate"], -row["calls"]))


def tool_metrics(runs: list[dict] | None = None) -> list[dict]:
    """Per-tool breakdown, most used first."""
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "ok": 0})
    for call in iter_tool_calls(runs):
        bucket = grouped[call.get("name", "unknown")]
        bucket["calls"] += 1
        bucket["ok"] += 1 if call.get("ok") else 0

    rows = [
        {
            "tool": name,
            "calls": stats["calls"],
            "successes": stats["ok"],
            "failures": stats["calls"] - stats["ok"],
            "success_rate": _rate(stats["ok"], stats["calls"]),
        }
        for name, stats in grouped.items()
    ]
    return sorted(rows, key=lambda row: -row["calls"])


def booking_metrics(runs: list[dict] | None = None) -> dict:
    """Booking success rate specifically, since the capstone names it.

    Attempts are counted from book_appointment calls only. find_available_slots
    is excluded - a slot search is not a booking attempt.
    """
    attempts = [call for call in iter_tool_calls(runs) if call.get("name") == "book_appointment"]
    confirmed = sum(1 for call in attempts if call.get("ok"))
    return {
        "attempts": len(attempts),
        "confirmed": confirmed,
        "failed": len(attempts) - confirmed,
        "success_rate": _rate(confirmed, len(attempts)),
    }


def planning_metrics(runs: list[dict] | None = None) -> dict:
    """How well the plan predicted the tools actually used.

    Precision: of the tools planned, how many were used.
    Recall:    of the tools used, how many were planned.
    Low recall means the agent improvised beyond its plan, which is worth
    seeing even though it is not automatically wrong.
    """
    runs = load_runs() if runs is None else runs
    scored = [run for run in runs if run.get("planned_tools")]
    if not scored:
        return {"runs_with_plan": 0, "llm_plans": 0, "heuristic_plans": 0,
                "avg_precision": 0.0, "avg_recall": 0.0, "avg_steps": 0.0}

    precisions: list[float] = []
    recalls: list[float] = []
    for run in scored:
        planned = set(run.get("planned_tools", []))
        used = set(run.get("tools_called", []))
        precisions.append(_rate(len(planned & used), len(planned)))
        recalls.append(_rate(len(planned & used), len(used)) if used else 0.0)

    return {
        "runs_with_plan": len(scored),
        "llm_plans": sum(1 for run in scored if run.get("plan_source") == "llm"),
        "heuristic_plans": sum(1 for run in scored if run.get("plan_source") == "heuristic"),
        "avg_precision": round(mean(precisions), 4),
        "avg_recall": round(mean(recalls), 4),
        "avg_steps": round(mean(run.get("planned_steps", 0) for run in scored), 2),
    }


def evaluation_metrics(evaluations: list[dict] | None = None) -> dict:
    """Grades from the QA evaluator, grouped by which capability was tested."""
    evaluations = load_evaluations() if evaluations is None else evaluations
    if not evaluations:
        return {"evaluated": 0, "correct": 0, "accuracy": 0.0, "by_category": []}

    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for record in evaluations:
        bucket = grouped[record.get("category", "uncategorised")]
        bucket["total"] += 1
        bucket["correct"] += 1 if record.get("correct") else 0

    correct = sum(1 for record in evaluations if record.get("correct"))
    return {
        "evaluated": len(evaluations),
        "correct": correct,
        "accuracy": _rate(correct, len(evaluations)),
        "by_category": sorted(
            (
                {
                    "category": category,
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "accuracy": _rate(stats["correct"], stats["total"]),
                }
                for category, stats in grouped.items()
            ),
            key=lambda row: row["accuracy"],
        ),
    }


def dashboard_snapshot() -> dict:
    """Everything the Streamlit metrics page needs, in one call."""
    runs = load_runs()
    return {
        "overall": overall_metrics(runs),
        "modules": module_metrics(runs),
        "tools": tool_metrics(runs),
        "booking": booking_metrics(runs),
        "planning": planning_metrics(runs),
        "evaluation": evaluation_metrics(),
    }
