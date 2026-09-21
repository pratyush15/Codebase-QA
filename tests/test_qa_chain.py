
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from core.qa_chain import run_qa_streaming


def _doc(source="app/core/ast_chunking.py"):
    return Document(
        page_content="def foo(): pass",
        metadata={"source": source, "language": "Python", "chunk_index": 0, "total_chunks": 1},
    )


def _fake_llm(stream_chunks=("Answer.",), rewrite_return=None):
    """A fake LLM: .invoke() is used for the rewrite call, .stream() for the answer."""
    llm = Mock()
    llm.invoke.return_value = rewrite_return
    llm.stream.return_value = iter(stream_chunks)
    return llm


def _drain(gen):
    events = list(gen)
    by_type = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)
    return by_type


def test_no_rewrite_call_when_chat_history_is_empty():
    llm = _fake_llm()
    with patch("core.qa_chain.get_llm", return_value=llm), \
         patch("core.qa_chain.retrieve_documents_advanced", return_value=[_doc()]) as mock_retrieve:
        events = _drain(run_qa_streaming(
            collection_name="proj", question="How does chunking work?", chat_history=None,
        ))

    # invoke() should never be called for rewriting since there's no history
    llm.invoke.assert_not_called()
    # retrieval got the original question, unmodified
    assert mock_retrieve.call_args.kwargs["query"] == "How does chunking work?"
    metrics = events["metrics"][0]["value"]
    assert metrics["query_rewritten"] is False
    assert metrics["rewritten_query"] is None


def test_rewrite_changes_the_query_used_for_retrieval():
    llm = _fake_llm(rewrite_return="How does the class-based chunking version work?")
    history = [
        {"role": "user", "content": "How does function-level AST chunking work?"},
        {"role": "assistant", "content": "It walks the tree at function boundaries."},
    ]

    with patch("core.qa_chain.get_llm", return_value=llm), \
         patch("core.qa_chain.retrieve_documents_advanced", return_value=[_doc()]) as mock_retrieve:
        events = _drain(run_qa_streaming(
            collection_name="proj", question="what about the class version?", chat_history=history,
        ))

    assert mock_retrieve.call_args.kwargs["query"] == "How does the class-based chunking version work?"
    metrics = events["metrics"][0]["value"]
    assert metrics["query_rewritten"] is True
    assert metrics["rewritten_query"] == "How does the class-based chunking version work?"

    # the *answer* prompt should still use the user's original phrasing
    prompt_sent = llm.stream.call_args[0][0]
    assert "what about the class version?" in prompt_sent


def test_rewrite_disabled_via_config_skips_rewrite_even_with_history():
    llm = _fake_llm(rewrite_return="should not be used")
    history = [{"role": "user", "content": "earlier question"}]

    with patch("core.qa_chain.get_llm", return_value=llm), \
         patch("core.qa_chain.ENABLE_QUERY_REWRITE", False), \
         patch("core.qa_chain.retrieve_documents_advanced", return_value=[_doc()]) as mock_retrieve:
        events = _drain(run_qa_streaming(
            collection_name="proj", question="what about that?", chat_history=history,
        ))

    llm.invoke.assert_not_called()
    assert mock_retrieve.call_args.kwargs["query"] == "what about that?"
    assert events["metrics"][0]["value"]["query_rewritten"] is False


def test_conversation_context_included_in_answer_prompt_even_without_rewrite_change():
    # rewrite returns the same question unchanged, but conversation context
    # should still be threaded into the answer prompt for continuity.
    llm = _fake_llm(rewrite_return="what about that?")
    history = [{"role": "user", "content": "How does hybrid retrieval work?"}]

    with patch("core.qa_chain.get_llm", return_value=llm), \
         patch("core.qa_chain.retrieve_documents_advanced", return_value=[_doc()]):
        _drain(run_qa_streaming(
            collection_name="proj", question="what about that?", chat_history=history,
        ))

    prompt_sent = llm.stream.call_args[0][0]
    assert "## Recent conversation" in prompt_sent
    assert "How does hybrid retrieval work?" in prompt_sent


def test_no_conversation_block_on_first_question():
    llm = _fake_llm()
    with patch("core.qa_chain.get_llm", return_value=llm), \
         patch("core.qa_chain.retrieve_documents_advanced", return_value=[_doc()]):
        _drain(run_qa_streaming(collection_name="proj", question="How does chunking work?"))

    prompt_sent = llm.stream.call_args[0][0]
    assert "## Recent conversation" not in prompt_sent


def test_error_event_when_no_docs_found_regardless_of_rewrite():
    llm = _fake_llm(rewrite_return="standalone question")
    with patch("core.qa_chain.get_llm", return_value=llm), \
         patch("core.qa_chain.retrieve_documents_advanced", return_value=[]):
        events = _drain(run_qa_streaming(
            collection_name="proj", question="what about that?",
            chat_history=[{"role": "user", "content": "earlier"}],
        ))

    assert "error" in events
    assert "token" not in events


# ---------- multi-query expansion ----------

def test_expansion_disabled_by_default_no_extra_llm_call():
    llm = _fake_llm()
    with patch("core.qa_chain.get_llm", return_value=llm), \
         patch("core.qa_chain.retrieve_documents_advanced", return_value=[_doc()]) as mock_retrieve:
        events = _drain(run_qa_streaming(
            collection_name="proj", question="how does login work?", retrieval_mode="hybrid_rerank",
        ))

    llm.invoke.assert_not_called()
    assert mock_retrieve.call_args.kwargs["extra_queries"] is None
    assert events["metrics"][0]["value"]["expanded_queries"] is None


def test_expansion_enabled_passes_paraphrases_to_retrieval():
    llm = _fake_llm()
    llm.invoke.return_value = "1. authenticate a user session\n2. validate login credentials"

    with patch("core.qa_chain.get_llm", return_value=llm), \
         patch("core.qa_chain.ENABLE_QUERY_EXPANSION", True), \
         patch("core.qa_chain.retrieve_documents_advanced", return_value=[_doc()]) as mock_retrieve:
        events = _drain(run_qa_streaming(
            collection_name="proj", question="how does login work?", retrieval_mode="hybrid_rerank",
        ))

    llm.invoke.assert_called_once()
    extra = mock_retrieve.call_args.kwargs["extra_queries"]
    assert extra == ["authenticate a user session", "validate login credentials"]
    metrics = events["metrics"][0]["value"]
    assert metrics["expanded_queries"] == extra


def test_expansion_skipped_for_vector_mode_even_when_enabled():
    llm = _fake_llm()
    with patch("core.qa_chain.get_llm", return_value=llm), \
         patch("core.qa_chain.ENABLE_QUERY_EXPANSION", True), \
         patch("core.qa_chain.retrieve_documents_advanced", return_value=[_doc()]) as mock_retrieve:
        events = _drain(run_qa_streaming(
            collection_name="proj", question="how does login work?", retrieval_mode="vector",
        ))

    llm.invoke.assert_not_called()
    assert mock_retrieve.call_args.kwargs["extra_queries"] is None
    assert events["metrics"][0]["value"]["expanded_queries"] is None


def test_expansion_and_rewrite_both_fire_independently():
    llm = _fake_llm()
    # first invoke() call is the rewrite, second is the expansion
    llm.invoke.side_effect = [
        "How does the authentication flow work end to end?",
        "1. validate login session\n2. check user credentials",
    ]
    history = [{"role": "user", "content": "earlier question about login"}]

    with patch("core.qa_chain.get_llm", return_value=llm), \
         patch("core.qa_chain.ENABLE_QUERY_EXPANSION", True), \
         patch("core.qa_chain.retrieve_documents_advanced", return_value=[_doc()]) as mock_retrieve:
        events = _drain(run_qa_streaming(
            collection_name="proj", question="what about the auth flow?",
            chat_history=history, retrieval_mode="hybrid",
        ))

    assert llm.invoke.call_count == 2
    assert mock_retrieve.call_args.kwargs["query"] == "How does the authentication flow work end to end?"
    assert mock_retrieve.call_args.kwargs["extra_queries"] == [
        "validate login session", "check user credentials",
    ]
    metrics = events["metrics"][0]["value"]
    assert metrics["query_rewritten"] is True
    assert metrics["expanded_queries"] == ["validate login session", "check user credentials"]


def test_expansion_failure_falls_back_to_none_without_breaking_query():
    llm = _fake_llm()
    llm.invoke.side_effect = RuntimeError("ollama unreachable")

    with patch("core.qa_chain.get_llm", return_value=llm), \
         patch("core.qa_chain.ENABLE_QUERY_EXPANSION", True), \
         patch("core.qa_chain.retrieve_documents_advanced", return_value=[_doc()]) as mock_retrieve:
        events = _drain(run_qa_streaming(
            collection_name="proj", question="how does login work?", retrieval_mode="hybrid",
        ))

    assert mock_retrieve.call_args.kwargs["extra_queries"] is None
    assert "token" in events  # answer generation still proceeds normally
    assert events["metrics"][0]["value"]["expanded_queries"] is None


# ---------- filter_symbol plumbing ----------

def test_filter_symbol_passed_through_to_retrieval():
    llm = _fake_llm()
    with patch("core.qa_chain.get_llm", return_value=llm), \
         patch("core.qa_chain.retrieve_documents_advanced", return_value=[_doc()]) as mock_retrieve:
        _drain(run_qa_streaming(
            collection_name="proj", question="how does X work?", filter_symbol="retrieve_documents",
        ))

    assert mock_retrieve.call_args.kwargs["filter_symbol"] == "retrieve_documents"


def test_filter_symbol_defaults_to_none():
    llm = _fake_llm()
    with patch("core.qa_chain.get_llm", return_value=llm), \
         patch("core.qa_chain.retrieve_documents_advanced", return_value=[_doc()]) as mock_retrieve:
        _drain(run_qa_streaming(collection_name="proj", question="how does X work?"))

    assert mock_retrieve.call_args.kwargs["filter_symbol"] is None


# ---------- k passthrough ----------

def test_k_defaults_to_max_retrieval_docs():
    from config import MAX_RETRIEVAL_DOCS

    llm = _fake_llm()
    with patch("core.qa_chain.get_llm", return_value=llm), \
         patch("core.qa_chain.retrieve_documents_advanced", return_value=[_doc()]) as mock_retrieve:
        _drain(run_qa_streaming(collection_name="proj", question="how does X work?"))

    assert mock_retrieve.call_args.kwargs["k"] == MAX_RETRIEVAL_DOCS


def test_k_override_passed_through_to_retrieval():
    llm = _fake_llm()
    with patch("core.qa_chain.get_llm", return_value=llm), \
         patch("core.qa_chain.retrieve_documents_advanced", return_value=[_doc()]) as mock_retrieve:
        _drain(run_qa_streaming(collection_name="proj", question="how does X work?", k=3))

    assert mock_retrieve.call_args.kwargs["k"] == 3


# ---------- query metrics logging (core.query_log) ----------

def test_metrics_logging_disabled_by_default_no_record_query_call():
    llm = _fake_llm()
    with patch("core.qa_chain.get_llm", return_value=llm), \
         patch("core.qa_chain.retrieve_documents_advanced", return_value=[_doc()]), \
         patch("core.qa_chain.record_query") as mock_record:
        _drain(run_qa_streaming(collection_name="proj", question="how does X work?"))

    # conftest.py sets ENABLE_QUERY_METRICS_LOG=false for the whole test
    # suite specifically so this is the default behavior under test.
    mock_record.assert_not_called()


def test_metrics_logging_records_successful_query_when_enabled():
    llm = _fake_llm()
    with patch("core.qa_chain.get_llm", return_value=llm), \
         patch("core.qa_chain.ENABLE_QUERY_METRICS_LOG", True), \
         patch("core.qa_chain.retrieve_documents_advanced", return_value=[_doc()]), \
         patch("core.qa_chain.record_query") as mock_record:
        _drain(run_qa_streaming(collection_name="proj", question="how does X work?", retrieval_mode="hybrid"))

    mock_record.assert_called_once()
    kwargs = mock_record.call_args.kwargs
    assert kwargs["collection"] == "proj"
    assert kwargs["retrieval_mode"] == "hybrid"
    assert kwargs["hit_count"] == 1
    assert kwargs["error"] is False


def test_metrics_logging_records_error_case_when_no_docs_found():
    llm = _fake_llm()
    with patch("core.qa_chain.get_llm", return_value=llm), \
         patch("core.qa_chain.ENABLE_QUERY_METRICS_LOG", True), \
         patch("core.qa_chain.retrieve_documents_advanced", return_value=[]), \
         patch("core.qa_chain.record_query") as mock_record:
        _drain(run_qa_streaming(collection_name="proj", question="how does X work?"))

    mock_record.assert_called_once()
    kwargs = mock_record.call_args.kwargs
    assert kwargs["error"] is True
    assert kwargs["hit_count"] == 0


def test_metrics_logging_passes_log_path_from_config():
    llm = _fake_llm()
    with patch("core.qa_chain.get_llm", return_value=llm), \
         patch("core.qa_chain.ENABLE_QUERY_METRICS_LOG", True), \
         patch("core.qa_chain.QUERY_METRICS_LOG_PATH", "/tmp/fake-metrics-path.jsonl"), \
         patch("core.qa_chain.retrieve_documents_advanced", return_value=[_doc()]), \
         patch("core.qa_chain.record_query") as mock_record:
        _drain(run_qa_streaming(collection_name="proj", question="how does X work?"))

    assert mock_record.call_args.args[0] == "/tmp/fake-metrics-path.jsonl"