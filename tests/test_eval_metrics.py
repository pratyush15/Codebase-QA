import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL_DIR))

from metrics import precision_at_k, recall_at_k, reciprocal_rank, evaluate_single, aggregate_results  # noqa: E402


# ---------- precision_at_k ----------

def test_precision_at_k_all_relevant():
    retrieved = ["a.py", "b.py"]
    expected = {"a.py", "b.py"}
    assert precision_at_k(retrieved, expected, k=2) == 1.0


def test_precision_at_k_none_relevant():
    retrieved = ["x.py", "y.py"]
    expected = {"a.py"}
    assert precision_at_k(retrieved, expected, k=2) == 0.0


def test_precision_at_k_partial():
    retrieved = ["a.py", "x.py", "y.py", "z.py"]
    expected = {"a.py"}
    assert precision_at_k(retrieved, expected, k=4) == 0.25


def test_precision_at_k_empty_retrieved():
    assert precision_at_k([], {"a.py"}, k=5) == 0.0


def test_precision_at_k_truncates_to_k():
    retrieved = ["a.py", "b.py", "irrelevant.py"]
    expected = {"a.py", "b.py"}
    # only first 2 considered — both relevant -> precision 1.0, not 2/3
    assert precision_at_k(retrieved, expected, k=2) == 1.0


# ---------- recall_at_k ----------

def test_recall_at_k_found():
    retrieved = ["x.py", "a.py", "y.py"]
    expected = {"a.py"}
    assert recall_at_k(retrieved, expected, k=3) == 1.0


def test_recall_at_k_not_found_within_k():
    retrieved = ["x.py", "y.py", "a.py"]
    expected = {"a.py"}
    assert recall_at_k(retrieved, expected, k=2) == 0.0


def test_recall_at_k_multiple_expected_partial():
    retrieved = ["a.py", "x.py"]
    expected = {"a.py", "b.py"}
    assert recall_at_k(retrieved, expected, k=2) == 0.5


def test_recall_at_k_empty_expected():
    assert recall_at_k(["a.py"], set(), k=5) == 0.0


# ---------- reciprocal_rank ----------

def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a.py", "b.py"], {"a.py"}) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["x.py", "y.py", "a.py"], {"a.py"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_not_found():
    assert reciprocal_rank(["x.py", "y.py"], {"a.py"}) == 0.0


# ---------- evaluate_single / aggregate_results ----------

def test_evaluate_single_hit_flag():
    result = evaluate_single(["a.py"], {"a.py"}, k=5)
    assert result["hit"] is True
    assert result["reciprocal_rank"] == 1.0


def test_evaluate_single_miss_flag():
    result = evaluate_single(["x.py"], {"a.py"}, k=5)
    assert result["hit"] is False
    assert result["reciprocal_rank"] == 0.0


def test_aggregate_results_averages_across_questions():
    results = [
        evaluate_single(["a.py"], {"a.py"}, k=5),   # perfect hit
        evaluate_single(["x.py"], {"a.py"}, k=5),   # miss
    ]
    agg = aggregate_results(results)
    assert agg["hit_rate"] == 0.5
    assert agg["n"] == 2


def test_aggregate_results_empty():
    agg = aggregate_results([])
    assert agg["n"] == 0
    assert agg["hit_rate"] == 0.0