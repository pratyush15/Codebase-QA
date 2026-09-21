
from unittest.mock import Mock

from core.query_rewrite import format_recent_turns, rewrite_query_standalone


# ---------- format_recent_turns ----------

def test_format_recent_turns_returns_empty_string_for_no_history():
    assert format_recent_turns(None) == ""
    assert format_recent_turns([]) == ""


def test_format_recent_turns_labels_roles():
    history = [
        {"role": "user", "content": "How does chunking work?"},
        {"role": "assistant", "content": "It uses tree-sitter."},
    ]
    rendered = format_recent_turns(history)
    assert "User: How does chunking work?" in rendered
    assert "Assistant: It uses tree-sitter." in rendered


def test_format_recent_turns_only_keeps_last_max_turns_pairs():
    # 5 (user, assistant) pairs; ask for only the last 2 pairs (4 messages).
    history = []
    for i in range(5):
        history.append({"role": "user", "content": f"question {i}"})
        history.append({"role": "assistant", "content": f"answer {i}"})

    rendered = format_recent_turns(history, max_turns=2)

    assert "question 3" in rendered
    assert "question 4" in rendered
    assert "question 0" not in rendered
    assert "question 1" not in rendered
    assert "question 2" not in rendered


def test_format_recent_turns_skips_empty_content():
    history = [{"role": "user", "content": "   "}, {"role": "assistant", "content": "real answer"}]
    rendered = format_recent_turns(history)
    assert rendered == "Assistant: real answer"


def test_format_recent_turns_truncates_long_content():
    history = [{"role": "assistant", "content": "x" * 1000}]
    rendered = format_recent_turns(history)
    assert len(rendered) < 1000
    assert rendered.endswith("...")


# ---------- rewrite_query_standalone ----------

def test_rewrite_returns_original_question_when_no_history():
    llm = Mock()
    result = rewrite_query_standalone("what about the class version?", None, llm)
    assert result == "what about the class version?"
    llm.invoke.assert_not_called()


def test_rewrite_returns_original_question_when_history_is_empty_list():
    llm = Mock()
    result = rewrite_query_standalone("what about the class version?", [], llm)
    assert result == "what about the class version?"
    llm.invoke.assert_not_called()


def test_rewrite_calls_llm_and_returns_rewritten_question():
    llm = Mock()
    llm.invoke.return_value = "How does the class-based chunking version of ast_chunking.py work?"

    history = [
        {"role": "user", "content": "How does function-level AST chunking work?"},
        {"role": "assistant", "content": "It walks the syntax tree and slices at function boundaries."},
    ]

    result = rewrite_query_standalone("what about the class version?", history, llm)

    assert result == "How does the class-based chunking version of ast_chunking.py work?"
    llm.invoke.assert_called_once()


def test_rewrite_strips_wrapping_quotes_from_llm_output():
    llm = Mock()
    llm.invoke.return_value = '"How does the reranker handle failures?"'
    history = [{"role": "user", "content": "Tell me about reranking."}]

    result = rewrite_query_standalone("what if it fails?", history, llm)

    assert result == "How does the reranker handle failures?"


def test_rewrite_falls_back_to_original_on_empty_llm_output():
    llm = Mock()
    llm.invoke.return_value = "   "
    history = [{"role": "user", "content": "Tell me about reranking."}]

    result = rewrite_query_standalone("what if it fails?", history, llm)

    assert result == "what if it fails?"


def test_rewrite_falls_back_to_original_on_llm_error():
    llm = Mock()
    llm.invoke.side_effect = RuntimeError("ollama unreachable")
    history = [{"role": "user", "content": "Tell me about reranking."}]

    result = rewrite_query_standalone("what if it fails?", history, llm)

    assert result == "what if it fails?"


def test_rewrite_respects_max_turns_when_building_history_prompt():
    llm = Mock()
    llm.invoke.return_value = "standalone question"

    history = []
    for i in range(5):
        history.append({"role": "user", "content": f"question {i}"})
        history.append({"role": "assistant", "content": f"answer {i}"})

    rewrite_query_standalone("follow up", history, llm, max_turns=1)

    prompt_sent = llm.invoke.call_args[0][0]
    assert "question 4" in prompt_sent
    assert "question 0" not in prompt_sent


# ---------- _parse_numbered_list ----------

from core.query_rewrite import _parse_numbered_list, expand_query


def test_parse_numbered_list_handles_standard_numbering():
    text = "1. authenticate a user\n2. validate login session\n"
    items = _parse_numbered_list(text, 2)
    assert items == ["authenticate a user", "validate login session"]


def test_parse_numbered_list_handles_parenthesis_numbering():
    text = "1) foo\n2) bar"
    assert _parse_numbered_list(text, 2) == ["foo", "bar"]


def test_parse_numbered_list_handles_bullet_dashes():
    text = "- foo\n- bar"
    assert _parse_numbered_list(text, 2) == ["foo", "bar"]


def test_parse_numbered_list_falls_back_to_plain_lines_when_unnumbered():
    text = "foo\nbar"
    assert _parse_numbered_list(text, 2) == ["foo", "bar"]


def test_parse_numbered_list_strips_wrapping_quotes():
    text = '1. "foo bar"'
    assert _parse_numbered_list(text, 1) == ["foo bar"]


def test_parse_numbered_list_respects_expected_count():
    text = "1. a\n2. b\n3. c\n4. d"
    assert _parse_numbered_list(text, 2) == ["a", "b"]


def test_parse_numbered_list_skips_blank_lines():
    text = "1. a\n\n2. b"
    assert _parse_numbered_list(text, 2) == ["a", "b"]


def test_parse_numbered_list_empty_text_returns_empty():
    assert _parse_numbered_list("", 3) == []


# ---------- expand_query ----------

def test_expand_query_returns_empty_for_n_zero_or_negative():
    llm = Mock()
    assert expand_query("how does auth work?", llm, n=0) == []
    assert expand_query("how does auth work?", llm, n=-1) == []
    llm.invoke.assert_not_called()


def test_expand_query_returns_paraphrases_from_llm():
    llm = Mock()
    llm.invoke.return_value = "1. authenticate a user session\n2. validate login credentials"

    result = expand_query("how does login work?", llm, n=2)

    assert result == ["authenticate a user session", "validate login credentials"]
    llm.invoke.assert_called_once()


def test_expand_query_falls_back_to_empty_on_llm_error():
    llm = Mock()
    llm.invoke.side_effect = RuntimeError("ollama unreachable")

    assert expand_query("how does login work?", llm, n=2) == []


def test_expand_query_deduplicates_against_original_question():
    llm = Mock()
    # first "paraphrase" is just the original question restated identically (case-insensitive)
    llm.invoke.return_value = "1. How does login work?\n2. validate login credentials"

    result = expand_query("how does login work?", llm, n=2)

    assert "How does login work?" not in result
    assert "validate login credentials" in result


def test_expand_query_deduplicates_paraphrases_against_each_other():
    llm = Mock()
    llm.invoke.return_value = "1. authenticate user\n2. Authenticate User\n3. check session token"

    result = expand_query("how does login work?", llm, n=3)

    # case-insensitive dedup should drop the near-identical second item
    assert len(result) == 2
    assert "check session token" in result


def test_expand_query_respects_n_even_with_more_candidates_returned():
    llm = Mock()
    llm.invoke.return_value = "1. a\n2. b\n3. c\n4. d"

    result = expand_query("original", llm, n=2)
    assert len(result) <= 2


def test_expand_query_empty_llm_output_returns_empty_list():
    llm = Mock()
    llm.invoke.return_value = ""
    assert expand_query("how does login work?", llm, n=2) == []