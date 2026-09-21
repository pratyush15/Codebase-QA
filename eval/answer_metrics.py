"""
Answer-quality metrics — the answer-generation half of the eval, on top of
metrics.py's retrieval-quality half.

Retrieval eval (metrics.py) only checks "did we find the right file?" — it
says nothing about whether the LLM actually used that file correctly to
produce a correct answer. These functions close that loop with a fast,
deterministic, no-LLM-call signal: whether specific facts/identifiers a
correct answer should mention actually appear in the generated text.

See eval/llm_judge.py for the complementary (opt-in, LLM-call-requiring)
answer-quality signal — LLM-as-judge grading, which catches a wrong or
hallucinated explanation that happens to still namedrop the right terms,
something keyword matching alone can't.

Pure functions — no dependency on Chroma, Ollama, or any live service — so
they're fully unit-testable and reusable from both the CLI harness
(eval/run_eval.py) and tests/test_eval_metrics.py, same as metrics.py.
"""
from typing import List


def keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
    """
    Fraction of expected_keywords found anywhere in answer (case-insensitive
    substring match). Returns 0.0 for an empty answer or an empty keyword
    list — an empty expectation list can't be "covered" by anything, so it
    doesn't silently count as a perfect score.
    """
    if not expected_keywords or not answer:
        return 0.0

    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def keyword_hit(answer: str, expected_keywords: List[str], min_coverage: float = 0.5) -> bool:
    """
    Whether the answer covers at least `min_coverage` of the expected
    keywords — a coarse pass/fail signal for aggregate reporting, analogous
    to metrics.py's recall_at_k -> hit_rate. Deliberately not 100%: a
    correct answer can reasonably paraphrase some, but not all, of the
    specific terms/identifiers a reference answer would use.
    """
    return keyword_coverage(answer, expected_keywords) >= min_coverage


def evaluate_answer(answer: str, expected_keywords: List[str], min_coverage: float = 0.5) -> dict:
    return {
        "keyword_coverage": keyword_coverage(answer, expected_keywords),
        "keyword_hit": keyword_hit(answer, expected_keywords, min_coverage),
    }


def aggregate_answer_results(per_question_results: List[dict]) -> dict:
    """Mean coverage + hit-rate across all evaluated questions."""
    if not per_question_results:
        return {"keyword_coverage": 0.0, "keyword_hit_rate": 0.0, "n": 0}

    n = len(per_question_results)
    return {
        "keyword_coverage": round(sum(r["keyword_coverage"] for r in per_question_results) / n, 4),
        "keyword_hit_rate": round(sum(1 for r in per_question_results if r["keyword_hit"]) / n, 4),
        "n": n,
    }