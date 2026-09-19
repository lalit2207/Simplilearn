"""Evaluate the agent against the golden dataset.

    python scripts/run_eval.py                          # all cases
    python scripts/run_eval.py --category medical_search
    python scripts/run_eval.py --case multi-01
    python scripts/run_eval.py --list

Requires GROQ_API_KEY. Results append to logs/evaluations.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings  # noqa: E402
from src.llmops import EVAL_CASES, cases_for, categories, evaluate_cases, summarise  # noqa: E402
from src.llmops.dataset import CASES_BY_ID  # noqa: E402
from src.llmops.metrics import module_metrics, overall_metrics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the healthcare agent.")
    parser.add_argument("--category", action="append", help="limit to a category (repeatable)")
    parser.add_argument("--case", action="append", help="run specific case ids (repeatable)")
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    parser.add_argument("--no-planner", action="store_true", help="use heuristic planning")
    args = parser.parse_args()

    if args.list:
        print(f"{len(EVAL_CASES)} cases across categories: {', '.join(categories())}\n")
        for case in EVAL_CASES:
            print(f"  {case.id:<12} [{case.category}]  {case.question[:70]}")
        return 0

    if args.case:
        selected = [CASES_BY_ID[cid] for cid in args.case if cid in CASES_BY_ID]
        unknown = [cid for cid in args.case if cid not in CASES_BY_ID]
        if unknown:
            print(f"Unknown case ids: {', '.join(unknown)}")
            return 1
    else:
        selected = cases_for(args.category)

    if not selected:
        print("No cases matched.")
        return 1

    if not settings.llm_ready:
        print("GROQ_API_KEY is not configured in .env - evaluation needs a live model.")
        return 1

    print(f"Evaluating {len(selected)} case(s) with {settings.groq_model}")
    print(f"Judge model: {settings.groq_eval_model}\n")

    results = evaluate_cases(selected, use_planner=not args.no_planner)

    for result in results:
        status = "PASS" if result.correct else ("ERROR" if result.error else "FAIL")
        print(f"[{status}] {result.case_id:<12} {result.category}")
        print(f"    grade={result.grade} keywords={result.keyword_coverage} tools={result.tool_coverage}")
        if result.missing_keywords:
            print(f"    missing keywords: {', '.join(result.missing_keywords)}")
        if result.missing_tools:
            print(f"    missing tools   : {', '.join(result.missing_tools)}")
        if result.error:
            print(f"    error: {result.error}")
        if result.answer:
            first_line = result.answer.strip().splitlines()[0]
            print(f"    answer: {first_line[:100]}")
        print()

    print("=" * 70)
    print("EVALUATION SUMMARY")
    for key, value in summarise(results).items():
        print(f"  {key:<22} {value}")

    print("\nRUN METRICS (all logged runs)")
    for key, value in overall_metrics().items():
        print(f"  {key:<22} {value}")

    print("\nPER-MODULE SUCCESS")
    for row in module_metrics():
        print(
            f"  {row['module']:<24} {row['successes']}/{row['calls']} "
            f"({row['success_rate']:.0%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
