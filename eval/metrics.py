"""
Retrieval evaluation metrics.

Pure functions — no dependency on Chroma, Ollama, or any live service — so
they're fully unit-testable and reusable from both the CLI harness
(eval/run_eval.py) and tests/test_eval_metrics.py.

Each metric takes `retrieved_sources` (an ordered list of file paths, as
returned by retriever.py's get_source_summary — best match first) and
`expected_sources` (the set of file paths a good answer should have drawn
from, from the hand-built eval set).
"""
from typing import List, Set


def precision_at_k(retrieved_sources: List[str], expected_sources: Set[str], k: int) -> float:
    """
    Of the top-k retrieved sources, what fraction are actually relevant?
    Measures how much noise is mixed in with the signal.
    """
    top_k = retrieved_sources[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for s in top_k if s in expected_sources)
    return relevant / len(top_k)


def recall_at_k(retrieved_sources: List[str], expected_sources: Set[str], k: int) -> float:
    """
    Of all the expected relevant sources, what fraction were found
    somewhere in the top-k? (Hit-rate when there's exactly one expected
    source per question, which is the common case in this eval set.)
    """
    if not expected_sources:
        return 0.0
    top_k_set = set(retrieved_sources[:k])
    found = len(expected_sources & top_k_set)
    return found / len(expected_sources)


def reciprocal_rank(retrieved_sources: List[str], expected_sources: Set[str]) -> float:
    """
    1 / (rank of the first relevant result), or 0.0 if none of the expected
    sources appear anywhere in retrieved_sources. Rewards ranking a
    relevant result near the top, not just including it somewhere.
    """
    for i, source in enumerate(retrieved_sources, start=1):
        if source in expected_sources:
            return 1.0 / i
    return 0.0


def evaluate_single(retrieved_sources: List[str], expected_sources: Set[str], k: int) -> dict:
    return {
        "precision_at_k": precision_at_k(retrieved_sources, expected_sources, k),
        "recall_at_k": recall_at_k(retrieved_sources, expected_sources, k),
        "reciprocal_rank": reciprocal_rank(retrieved_sources, expected_sources),
        "hit": recall_at_k(retrieved_sources, expected_sources, k) > 0,
    }


def aggregate_results(per_question_results: List[dict]) -> dict:
    """Mean each metric across all evaluated questions."""
    if not per_question_results:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "hit_rate": 0.0, "n": 0}

    n = len(per_question_results)
    return {
        "precision_at_k": round(sum(r["precision_at_k"] for r in per_question_results) / n, 4),
        "recall_at_k": round(sum(r["recall_at_k"] for r in per_question_results) / n, 4),
        "mrr": round(sum(r["reciprocal_rank"] for r in per_question_results) / n, 4),
        "hit_rate": round(sum(1 for r in per_question_results if r["hit"]) / n, 4),
        "n": n,
    }