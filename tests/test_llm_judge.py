import sys
from pathlib import Path
from unittest.mock import Mock

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL_DIR))

from llm_judge import (  # noqa: E402
    judge_answer,
    aggregate_judge_results,
    _parse_judge_response,
    VERDICT_SCORES,
)


# ---------- _parse_judge_response ----------

def test_parse_judge_response_pure_json():
    result = _parse_judge_response('{"verdict": "correct", "reasoning": "spot on"}')
    assert result == {"verdict": "correct", "reasoning": "spot on"}


def test_parse_judge_response_json_wrapped_in_prose():
    raw = 'Sure, here is my grade:\n{"verdict": "partial", "reasoning": "close but vague"}\nHope that helps!'
    result = _parse_judge_response(raw)
    assert result == {"verdict": "partial", "reasoning": "close but vague"}


def test_parse_judge_response_json_in_code_fence():
    raw = '```json\n{"verdict": "incorrect", "reasoning": "wrong file"}\n```'
    result = _parse_judge_response(raw)
    assert result == {"verdict": "incorrect", "reasoning": "wrong file"}


def test_parse_judge_response_no_json_returns_none():
    assert _parse_judge_response("I think this answer is correct.") is None


def test_parse_judge_response_malformed_json_returns_none():
    assert _parse_judge_response('{"verdict": "correct", "reasoning": }') is None


def test_parse_judge_response_empty_string_returns_none():
    assert _parse_judge_response("") is None


# ---------- judge_answer ----------

def test_judge_answer_correct_verdict():
    llm = Mock()
    llm.invoke.return_value = '{"verdict": "correct", "reasoning": "accurate and complete"}'

    result = judge_answer("How is X computed?", "X is computed via sha256.", llm)

    assert result["verdict"] == "correct"
    assert result["score"] == 1.0
    assert result["reasoning"] == "accurate and complete"


def test_judge_answer_partial_verdict():
    llm = Mock()
    llm.invoke.return_value = '{"verdict": "partial", "reasoning": "vague"}'

    result = judge_answer("q", "a", llm)

    assert result["verdict"] == "partial"
    assert result["score"] == 0.5


def test_judge_answer_incorrect_verdict():
    llm = Mock()
    llm.invoke.return_value = '{"verdict": "incorrect", "reasoning": "wrong"}'

    result = judge_answer("q", "a", llm)

    assert result["verdict"] == "incorrect"
    assert result["score"] == 0.0


def test_judge_answer_verdict_case_insensitive():
    llm = Mock()
    llm.invoke.return_value = '{"verdict": "CORRECT", "reasoning": "yes"}'

    result = judge_answer("q", "a", llm)

    assert result["verdict"] == "correct"
    assert result["score"] == 1.0


def test_judge_answer_llm_call_failure_returns_error_verdict():
    llm = Mock()
    llm.invoke.side_effect = RuntimeError("ollama unreachable")

    result = judge_answer("q", "a", llm)

    assert result["verdict"] == "error"
    assert result["score"] == 0.0
    assert "ollama unreachable" in result["reasoning"]


def test_judge_answer_unparseable_response_returns_error_verdict():
    llm = Mock()
    llm.invoke.return_value = "I refuse to answer in JSON."

    result = judge_answer("q", "a", llm)

    assert result["verdict"] == "error"
    assert result["score"] == 0.0


def test_judge_answer_unrecognized_verdict_string_returns_error():
    llm = Mock()
    llm.invoke.return_value = '{"verdict": "excellent", "reasoning": "great job"}'

    result = judge_answer("q", "a", llm)

    assert result["verdict"] == "error"
    assert result["score"] == 0.0


def test_judge_answer_missing_reasoning_defaults_to_empty_string():
    llm = Mock()
    llm.invoke.return_value = '{"verdict": "correct"}'

    result = judge_answer("q", "a", llm)

    assert result["verdict"] == "correct"
    assert result["reasoning"] == ""


def test_judge_answer_never_raises_on_llm_failure():
    # Explicit smoke test of the "never raises" contract stated in the docstring.
    llm = Mock()
    llm.invoke.side_effect = Exception("anything")
    judge_answer("q", "a", llm)  # must not raise


# ---------- VERDICT_SCORES sanity ----------

def test_verdict_scores_cover_all_three_verdicts():
    assert set(VERDICT_SCORES.keys()) == {"correct", "partial", "incorrect"}
    assert VERDICT_SCORES["correct"] == 1.0
    assert VERDICT_SCORES["incorrect"] == 0.0


# ---------- aggregate_judge_results ----------

def test_aggregate_judge_results_empty_list():
    result = aggregate_judge_results([])
    assert result["n"] == 0
    assert result["avg_judge_score"] == 0.0


def test_aggregate_judge_results_averages_score_and_counts_verdicts():
    per_question = [
        {"verdict": "correct", "score": 1.0, "reasoning": ""},
        {"verdict": "partial", "score": 0.5, "reasoning": ""},
        {"verdict": "incorrect", "score": 0.0, "reasoning": ""},
        {"verdict": "correct", "score": 1.0, "reasoning": ""},
    ]
    result = aggregate_judge_results(per_question)

    assert result["n"] == 4
    assert result["avg_judge_score"] == round((1.0 + 0.5 + 0.0 + 1.0) / 4, 4)
    assert result["correct"] == 2
    assert result["partial"] == 1
    assert result["incorrect"] == 1
    assert result["error"] == 0


def test_aggregate_judge_results_counts_error_verdicts_too():
    per_question = [
        {"verdict": "error", "score": 0.0, "reasoning": "judge call failed"},
        {"verdict": "correct", "score": 1.0, "reasoning": ""},
    ]
    result = aggregate_judge_results(per_question)

    assert result["error"] == 1
    assert result["correct"] == 1