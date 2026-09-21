import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL_DIR))

from answer_metrics import (  # noqa: E402
    keyword_coverage,
    keyword_hit,
    evaluate_answer,
    aggregate_answer_results,
)


# ---------- keyword_coverage ----------

def test_keyword_coverage_all_present():
    answer = "The hash is computed with hashlib.sha256 in hash_file_content."
    assert keyword_coverage(answer, ["hash_file_content", "sha256", "hashlib"]) == 1.0


def test_keyword_coverage_none_present():
    answer = "This code does something else entirely."
    assert keyword_coverage(answer, ["hash_file_content", "sha256"]) == 0.0


def test_keyword_coverage_partial():
    answer = "It uses sha256 to hash the content."
    assert keyword_coverage(answer, ["hash_file_content", "sha256", "hashlib"]) == 1 / 3


def test_keyword_coverage_case_insensitive():
    answer = "It uses SHA256 for hashing."
    assert keyword_coverage(answer, ["sha256"]) == 1.0


def test_keyword_coverage_empty_answer_is_zero():
    assert keyword_coverage("", ["sha256"]) == 0.0


def test_keyword_coverage_empty_keyword_list_is_zero_not_perfect():
    # An empty expectation list shouldn't silently count as "fully covered".
    assert keyword_coverage("anything at all", []) == 0.0


def test_keyword_coverage_substring_match_counts():
    # "hash" is a substring of "hashlib" and "hashing" — this is
    # deliberately loose (substring, not whole-word) matching.
    answer = "The function uses hashlib for hashing."
    assert keyword_coverage(answer, ["hash"]) == 1.0


# ---------- keyword_hit ----------

def test_keyword_hit_true_at_default_threshold():
    answer = "Uses sha256 and hashlib for hash_file_content."
    assert keyword_hit(answer, ["hash_file_content", "sha256", "hashlib"]) is True


def test_keyword_hit_false_below_default_threshold():
    answer = "Uses sha256 only."
    # 1/3 coverage < the default 0.5 threshold
    assert keyword_hit(answer, ["hash_file_content", "sha256", "hashlib"]) is False


def test_keyword_hit_exact_at_threshold_boundary_counts_as_hit():
    answer = "Uses sha256 and hashlib."
    # 2/3 >= 0.5 -> hit
    assert keyword_hit(answer, ["hash_file_content", "sha256", "hashlib"], min_coverage=0.5) is True


def test_keyword_hit_respects_custom_threshold():
    answer = "Uses sha256 only."
    assert keyword_hit(answer, ["hash_file_content", "sha256", "hashlib"], min_coverage=0.3) is True
    assert keyword_hit(answer, ["hash_file_content", "sha256", "hashlib"], min_coverage=0.9) is False


# ---------- evaluate_answer ----------

def test_evaluate_answer_returns_both_fields():
    answer = "Uses sha256 and hashlib for hash_file_content."
    result = evaluate_answer(answer, ["hash_file_content", "sha256", "hashlib"])
    assert result["keyword_coverage"] == 1.0
    assert result["keyword_hit"] is True


# ---------- aggregate_answer_results ----------

def test_aggregate_answer_results_empty_list():
    result = aggregate_answer_results([])
    assert result == {"keyword_coverage": 0.0, "keyword_hit_rate": 0.0, "n": 0}


def test_aggregate_answer_results_averages_and_counts_hits():
    per_question = [
        {"keyword_coverage": 1.0, "keyword_hit": True},
        {"keyword_coverage": 0.0, "keyword_hit": False},
        {"keyword_coverage": 0.5, "keyword_hit": True},
    ]
    result = aggregate_answer_results(per_question)
    assert result["keyword_coverage"] == round((1.0 + 0.0 + 0.5) / 3, 4)
    assert result["keyword_hit_rate"] == round(2 / 3, 4)
    assert result["n"] == 3