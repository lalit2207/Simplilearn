"""LLMOps: structured logging, metrics, and model evaluation."""

from src.llmops.dataset import EVAL_CASES, EvalCase, cases_for, categories
from src.llmops.evaluation import CaseResult, evaluate_cases, summarise
from src.llmops.metrics import (
    booking_metrics,
    dashboard_snapshot,
    evaluation_metrics,
    module_metrics,
    overall_metrics,
    planning_metrics,
    tool_metrics,
)
from src.llmops.run_log import (
    clear_logs,
    load_evaluations,
    load_runs,
    log_evaluation,
    log_run,
    module_for,
)

__all__ = [
    "EVAL_CASES",
    "CaseResult",
    "EvalCase",
    "booking_metrics",
    "cases_for",
    "categories",
    "clear_logs",
    "dashboard_snapshot",
    "evaluate_cases",
    "evaluation_metrics",
    "load_evaluations",
    "load_runs",
    "log_evaluation",
    "log_run",
    "module_for",
    "module_metrics",
    "overall_metrics",
    "planning_metrics",
    "summarise",
    "tool_metrics",
]
