"""Run the agent against a request from the command line.

    python scripts/run_agent.py
    python scripts/run_agent.py "Book a cardiologist for my mother next week"
    python scripts/run_agent.py --plan-only

The default question is the capstone's own sample scenario.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import HealthcareAgent, build_session, create_plan  # noqa: E402
from src.agent.memory import build_memory_section  # noqa: E402
from src.config import settings  # noqa: E402

SAMPLE_QUESTION = (
    "My 70-year-old father has chronic kidney disease. I want to book a "
    "nephrologist for him. Also, can you summarize latest treatment methods?"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the healthcare agent.")
    parser.add_argument("question", nargs="?", default=SAMPLE_QUESTION)
    parser.add_argument("--attendant-id", type=int, default=1)
    parser.add_argument(
        "--plan-only", action="store_true", help="show the plan without executing tools"
    )
    parser.add_argument(
        "--no-planner", action="store_true", help="use the heuristic plan instead of the LLM"
    )
    args = parser.parse_args()

    session = build_session(attendant_id=args.attendant_id)

    print("=" * 78)
    print(f"QUESTION: {args.question}")
    print(f"MODEL   : {settings.groq_model}   |   LLM configured: {settings.llm_ready}")
    print("=" * 78)

    if args.plan_only:
        context = build_memory_section(session, args.question, None)
        plan = create_plan(args.question, context=context, use_llm=not args.no_planner)
        print(f"\nPLAN (source: {plan.source})")
        print(f"Goal: {plan.goal}")
        for step in plan.steps:
            print(f"  {step.step}. {step.sub_goal}")
            print(f"     tool: {step.tool}")
            if step.why:
                print(f"     why : {step.why}")
        return 0

    trace = HealthcareAgent(session).run(args.question, use_planner=not args.no_planner)

    if trace.plan:
        print(f"\nPLAN (source: {trace.plan.source})")
        for step in trace.plan.steps:
            print(f"  {step.step}. {step.sub_goal}  [tool: {step.tool}]")

    print(f"\nTOOL CALLS ({len(trace.tool_calls)})")
    for index, call in enumerate(trace.tool_calls, start=1):
        flag = "ok" if call.ok else "FAILED"
        print(f"  {index}. {call.name} [{flag}]  args={call.arguments}")
        for line in call.preview.splitlines()[:4]:
            print(f"       {line}")

    print("\n" + "=" * 78)
    if trace.error:
        print(f"ERROR: {trace.error}")
    else:
        print("FINAL ANSWER")
        print("=" * 78)
        print(trace.answer)

    print("=" * 78)
    print(
        f"patient_id={trace.patient_id} | tools={len(trace.tool_calls)} "
        f"| failures={len(trace.failed_calls)} | {trace.duration_ms} ms"
    )
    return 0 if trace.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
