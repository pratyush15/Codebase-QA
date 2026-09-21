"""
LLM-as-judge: a second LLM call grades a generated answer against the
question it was answering, on a simple 3-way verdict scale. Complements
eval/answer_metrics.py's keyword checks — catches a wrong or hallucinated
explanation that happens to still mention the right identifiers/terms,
which pure keyword matching can't tell apart from an actually-correct one.

Opt-in (run_eval.py's --judge flag): every call here is a real LLM
generation, so it roughly doubles per-question eval cost on top of
answer-generation itself, and adds the judge model's own noise (it can
grade the same answer slightly differently across runs, being an LLM
call itself) rather than a fully deterministic signal like the keyword
checks.
"""
import json
import logging
from typing import List, Optional

from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


JUDGE_PROMPT = PromptTemplate(
    input_variables=["question", "answer"],
    template="""You are grading whether an AI assistant's answer about a codebase is factually correct and actually addresses the question asked.

Grade the "Answer" to the "Question" below as one of:
  - "correct": factually accurate and directly addresses the question
  - "partial": on the right topic but incomplete, vague, or only partly accurate
  - "incorrect": wrong, hallucinated, or doesn't address the question

Respond with ONLY a JSON object, no other text: {{"verdict": "correct"|"partial"|"incorrect", "reasoning": "<one sentence>"}}

## Question
{question}

## Answer
{answer}

## Grade (JSON only)
""",
)

VERDICT_SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def _parse_judge_response(raw: str) -> Optional[dict]:
    """
    Extracts a JSON object from the judge's raw output. A model asked for
    "JSON only" still sometimes wraps it in a sentence or a code fence —
    this pulls out the first {...} block rather than requiring the whole
    response to be pure JSON, and returns None (not a partial/garbage
    dict) if no valid JSON object can be found at all.
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None


def judge_answer(question: str, answer: str, llm) -> dict:
    """
    Returns {"verdict": "correct"|"partial"|"incorrect"|"error",
    "score": float, "reasoning": str}.

    Never raises: an LLM call failure or an unparseable/unrecognized
    response produces verdict="error", score=0.0, with the failure reason
    in "reasoning" — a broken judge call shows up as a visibly-wrong row
    in the eval report rather than crashing the whole eval run partway
    through a (potentially long, all-LLM-calls) evaluation.
    """
    prompt_text = JUDGE_PROMPT.format(question=question, answer=answer)

    try:
        raw = llm.invoke(prompt_text)
    except Exception as e:
        logger.exception("LLM judge call failed")
        return {"verdict": "error", "score": 0.0, "reasoning": f"judge call failed: {e}"}

    parsed = _parse_judge_response(raw or "")
    if parsed is None:
        return {"verdict": "error", "score": 0.0, "reasoning": f"unparseable judge response: {raw!r}"}

    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in VERDICT_SCORES:
        return {"verdict": "error", "score": 0.0, "reasoning": f"unrecognized verdict: {verdict!r}"}

    return {
        "verdict": verdict,
        "score": VERDICT_SCORES[verdict],
        "reasoning": str(parsed.get("reasoning", "")),
    }


def aggregate_judge_results(per_question_results: List[dict]) -> dict:
    """Mean judge score + per-verdict counts across all evaluated questions."""
    if not per_question_results:
        return {"avg_judge_score": 0.0, "correct": 0, "partial": 0, "incorrect": 0, "error": 0, "n": 0}

    n = len(per_question_results)
    counts = {"correct": 0, "partial": 0, "incorrect": 0, "error": 0}
    for r in per_question_results:
        verdict = r.get("verdict", "error")
        counts[verdict] = counts.get(verdict, 0) + 1

    return {
        "avg_judge_score": round(sum(r["score"] for r in per_question_results) / n, 4),
        **counts,
        "n": n,
    }