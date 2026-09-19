"""Model evaluation using QAEvalChain plus deterministic checks.

Three independent signals per case, because no single one is trustworthy:

  1. keyword coverage - deterministic, free, catches omitted facts
  2. tool coverage    - did the agent actually call what the task required
  3. QAEvalChain      - an LLM judges the answer against a reference

QAEvalChain grades semantic equivalence, so it correctly passes an answer that
is worded differently from the reference. What it will not reliably catch is a
fluent answer that quietly drops a required number, which is what check 1 is
for. Tool coverage catches the opposite failure: a plausible answer produced
without ever consulting the data, which both other checks can miss.

`QAEvalChain` moved to the `langchain_classic` package in LangChain 1.x.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from langchain_classic.evaluation.qa import QAEvalChain

from src.agent.executor import HealthcareAgent, build_session
from src.agent.llm import MissingAPIKeyError, get_eval_llm
from src.llmops.dataset import EVAL_CASES, EvalCase
from src.llmops.run_log import log_evaluation, log_run


@dataclass
class CaseResult:
    case_id: str
    category: str
    question: str
    reference: str
    answer: str
    grade: str = "UNKNOWN"          # CORRECT | INCORRECT | UNKNOWN
    correct: bool = False
    keyword_coverage: float = 0.0
    missing_keywords: list[str] = field(default_factory=list)
    tool_coverage: float = 0.0
    missing_tools: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    duration_ms: int = 0
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def keyword_coverage(answer: str, required: list[str]) -> tuple[float, list[str]]:
    """Fraction of required strings present, case-insensitively."""
    if not required:
        return 1.0, []
    lowered = (answer or "").lower()
    missing = [term for term in required if term.lower() not in lowered]
    return round((len(required) - len(missing)) / len(required), 4), missing


def tool_coverage(tools_called: list[str], required: list[str]) -> tuple[float, list[str]]:
    """Fraction of required tools the agent actually invoked."""
    if not required:
        return 1.0, []
    called = set(tools_called)
    missing = [tool for tool in required if tool not in called]
    return round((len(required) - len(missing)) / len(required), 4), missing


def grade_with_qa_eval_chain(items: list[dict], predictions: list[dict]) -> list[dict]:
    """Run QAEvalChain over question/reference/prediction triples.

    `items` need `query` and `answer` (the reference); `predictions` need
    `result`. Returns one graded dict per item, or empty on failure so the
    deterministic checks still stand on their own.
    """
    if not items:
        return []
    try:
        chain = QAEvalChain.from_llm(llm=get_eval_llm())
        return chain.evaluate(items, predictions, question_key="query", prediction_key="result")
    except MissingAPIKeyError:
        return []
    except Exception:
        return []


def _parse_grade(graded: dict | None) -> tuple[str, bool]:
    """Normalise the grader's free-text verdict.

    The judge may answer "CORRECT", "GRADE: CORRECT", or lowercase. Checking
    for INCORRECT first matters, since "CORRECT" is a substring of it.
    """
    if not graded:
        return "UNKNOWN", False
    raw = str(graded.get("results") or graded.get("text") or "").strip().upper()
    if "INCORRECT" in raw:
        return "INCORRECT", False
    if "CORRECT" in raw:
        return "CORRECT", True
    return "UNKNOWN", False


def evaluate_cases(
    cases: list[EvalCase] | None = None,
    attendant_id: int = 1,
    use_planner: bool = True,
    persist: bool = True,
) -> list[CaseResult]:
    """Run the agent over each case, then grade the answers.

    Every case runs in a fresh session so results are independent - shared
    memory would let one case's answer leak into the next.
    """
    cases = cases if cases is not None else EVAL_CASES
    results: list[CaseResult] = []
    graded_inputs: list[dict] = []
    graded_predictions: list[dict] = []

    for case in cases:
        session = build_session(attendant_id=attendant_id, thread_id=f"eval-{case.id}-{int(time.time())}")
        trace = HealthcareAgent(session).run(case.question, use_planner=use_planner)

        keyword_score, missing_keywords = keyword_coverage(trace.answer, case.must_include)
        tool_score, missing_tools = tool_coverage(trace.tools_used, case.needs_tools)

        results.append(
            CaseResult(
                case_id=case.id,
                category=case.category,
                question=case.question,
                reference=case.reference,
                answer=trace.answer,
                keyword_coverage=keyword_score,
                missing_keywords=missing_keywords,
                tool_coverage=tool_score,
                missing_tools=missing_tools,
                tools_called=trace.tools_used,
                duration_ms=trace.duration_ms,
                error=trace.error,
            )
        )

        if persist:
            log_run(trace, session_id=session.thread_id, extra={"eval_case": case.id})

        if trace.answer:
            graded_inputs.append({"query": case.question, "answer": case.reference})
            graded_predictions.append({"result": trace.answer})

    grades = grade_with_qa_eval_chain(graded_inputs, graded_predictions)

    # Grades come back only for cases that produced an answer, in order.
    answered = [result for result in results if result.answer]
    for result, graded in zip(answered, grades):
        result.grade, result.correct = _parse_grade(graded)

    if persist:
        for result in results:
            log_evaluation(result.as_dict())

    return results


def summarise(results: list[CaseResult]) -> dict:
    """Roll case results into headline numbers."""
    if not results:
        return {"cases": 0, "correct": 0, "accuracy": 0.0,
                "avg_keyword_coverage": 0.0, "avg_tool_coverage": 0.0, "errors": 0}

    total = len(results)
    return {
        "cases": total,
        "correct": sum(1 for r in results if r.correct),
        "accuracy": round(sum(1 for r in results if r.correct) / total, 4),
        "avg_keyword_coverage": round(sum(r.keyword_coverage for r in results) / total, 4),
        "avg_tool_coverage": round(sum(r.tool_coverage for r in results) / total, 4),
        "errors": sum(1 for r in results if r.error),
        "ungraded": sum(1 for r in results if r.grade == "UNKNOWN"),
    }
