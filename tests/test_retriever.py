
from langchain_core.documents import Document

from core.retriever import _build_filter, assemble_context, get_source_summary


# ---------- _build_filter ----------

def test_build_filter_returns_none_when_no_filters():
    assert _build_filter(None, None) is None


def test_build_filter_single_language_condition():
    result = _build_filter("Python", None)
    assert result == {"language": {"$eq": "Python"}}


def test_build_filter_single_source_condition():
    result = _build_filter(None, "app/main.py")
    assert result == {"source": {"$eq": "app/main.py"}}


def test_build_filter_combines_both_with_and():
    result = _build_filter("Python", "app/main.py")
    assert result == {
        "$and": [
            {"language": {"$eq": "Python"}},
            {"source": {"$eq": "app/main.py"}},
        ]
    }


# ---------- assemble_context ----------

def test_assemble_context_joins_chunks_with_separator():
    docs = [
        Document(page_content="chunk one", metadata={"source": "a.py", "chunk_index": 0}),
        Document(page_content="chunk two", metadata={"source": "a.py", "chunk_index": 1}),
    ]
    context = assemble_context(docs)
    assert "chunk one" in context
    assert "chunk two" in context
    assert "---" in context


def test_assemble_context_dedupes_identical_source_and_chunk_index():
    # Same (source, chunk_index) retrieved twice (e.g. via overlapping filters)
    # should only appear once in the assembled context.
    docs = [
        Document(page_content="dup chunk", metadata={"source": "a.py", "chunk_index": 0}),
        Document(page_content="dup chunk", metadata={"source": "a.py", "chunk_index": 0}),
        Document(page_content="unique chunk", metadata={"source": "b.py", "chunk_index": 0}),
    ]
    context = assemble_context(docs)
    assert context.count("dup chunk") == 1
    assert "unique chunk" in context


# ---------- get_source_summary ----------

def test_get_source_summary_dedupes_by_source():
    docs = [
        Document(page_content="c1", metadata={"source": "a.py", "language": "Python", "chunk_index": 0, "total_chunks": 3}),
        Document(page_content="c2", metadata={"source": "a.py", "language": "Python", "chunk_index": 1, "total_chunks": 3}),
        Document(page_content="c3", metadata={"source": "b.py", "language": "Python", "chunk_index": 0, "total_chunks": 1}),
    ]
    summary = get_source_summary(docs)
    sources = [s["source"] for s in summary]

    assert sources == ["a.py", "b.py"]  # a.py appears once despite 2 chunks


def test_get_source_summary_handles_missing_metadata_gracefully():
    docs = [Document(page_content="c1", metadata={})]
    summary = get_source_summary(docs)
    assert summary[0]["source"] == "unknown"


def test_get_source_summary_includes_symbol_fields():
    docs = [
        Document(page_content="c1", metadata={
            "source": "app/core/retriever.py", "chunk_index": 0,
            "symbol_name": "retrieve_documents", "symbol_type": "function", "parent_symbol": "",
        }),
    ]
    summary = get_source_summary(docs)
    assert summary[0]["symbol_name"] == "retrieve_documents"
    assert summary[0]["symbol_type"] == "function"
    assert summary[0]["parent_symbol"] == ""


def test_get_source_summary_symbol_fields_default_to_empty_string():
    docs = [Document(page_content="c1", metadata={"source": "a.py", "chunk_index": 0})]
    summary = get_source_summary(docs)
    assert summary[0]["symbol_name"] == ""
    assert summary[0]["symbol_type"] == ""
    assert summary[0]["parent_symbol"] == ""

# ---------- retrieve_documents_advanced: multi-query wiring ----------

from unittest.mock import patch
from core.retriever import retrieve_documents_advanced


def _doc(source, chunk_index=0):
    return Document(page_content="x", metadata={"source": source, "chunk_index": chunk_index})


@patch("core.hybrid_retriever.hybrid_retrieve_multi_query")
@patch("core.hybrid_retriever.hybrid_retrieve")
def test_hybrid_mode_without_extra_queries_uses_single_query_path(mock_single, mock_multi):
    mock_single.return_value = [_doc("a.py")]

    retrieve_documents_advanced("proj", "how does auth work?", mode="hybrid")

    mock_single.assert_called_once()
    mock_multi.assert_not_called()
    assert mock_single.call_args.args[1] == "how does auth work?"


@patch("core.hybrid_retriever.hybrid_retrieve_multi_query")
@patch("core.hybrid_retriever.hybrid_retrieve")
def test_hybrid_mode_with_extra_queries_uses_multi_query_path(mock_single, mock_multi):
    mock_multi.return_value = [_doc("a.py")]

    retrieve_documents_advanced(
        "proj", "how does auth work?", mode="hybrid",
        extra_queries=["authenticate a user", "check login session"],
    )

    mock_multi.assert_called_once()
    mock_single.assert_not_called()
    passed_queries = mock_multi.call_args.args[1]
    assert passed_queries == [
        "how does auth work?", "authenticate a user", "check login session",
    ]


@patch("core.hybrid_retriever.hybrid_retrieve_multi_query")
@patch("core.hybrid_retriever.hybrid_retrieve")
def test_empty_extra_queries_list_uses_single_query_path(mock_single, mock_multi):
    mock_single.return_value = [_doc("a.py")]

    retrieve_documents_advanced("proj", "how does auth work?", mode="hybrid", extra_queries=[])

    mock_single.assert_called_once()
    mock_multi.assert_not_called()


@patch("core.hybrid_retriever.hybrid_retrieve_multi_query")
@patch("core.hybrid_retriever.hybrid_retrieve")
def test_extra_queries_ignored_for_vector_mode(mock_single, mock_multi):
    # extra_queries only makes sense for the RRF-fusion hybrid paths.
    retrieve_documents_advanced(
        "proj", "how does auth work?", mode="vector", extra_queries=["a", "b"],
    )
    mock_single.assert_not_called()
    mock_multi.assert_not_called()


@patch("core.reranker.is_reranking_available", return_value=False)
@patch("core.hybrid_retriever.hybrid_retrieve_multi_query")
def test_hybrid_rerank_mode_with_extra_queries_still_fuses_multi_query(mock_multi, mock_available):
    mock_multi.return_value = [_doc("a.py"), _doc("b.py")]

    docs = retrieve_documents_advanced(
        "proj", "how does auth work?", mode="hybrid_rerank", extra_queries=["log a user in"],
    )

    mock_multi.assert_called_once()
    assert len(docs) == 2


def test_build_filter_single_symbol_condition():
    result = _build_filter(None, None, "retrieve_documents")
    assert result == {"symbol_name": {"$eq": "retrieve_documents"}}


def test_build_filter_symbol_combines_with_source_and_language():
    result = _build_filter("Python", "app/core/retriever.py", "retrieve_documents")
    assert result == {
        "$and": [
            {"language": {"$eq": "Python"}},
            {"source": {"$eq": "app/core/retriever.py"}},
            {"symbol_name": {"$eq": "retrieve_documents"}},
        ]
    }


def test_build_filter_symbol_alone_with_no_language_or_source():
    result = _build_filter(None, None, "foo")
    assert result == {"symbol_name": {"$eq": "foo"}}


# ---------- filter_symbol plumbing ----------

from unittest.mock import Mock
from core.retriever import retrieve_documents, retrieve_documents_mmr


@patch("core.retriever.get_vectorstore_for_query")
def test_retrieve_documents_passes_symbol_filter_to_similarity_search(mock_get_vs):
    mock_vs = Mock()
    mock_vs.similarity_search.return_value = []
    mock_get_vs.return_value = mock_vs

    retrieve_documents("proj", "how does X work?", filter_symbol="retrieve_documents")

    kwargs = mock_vs.similarity_search.call_args.kwargs
    assert kwargs["filter"] == {"symbol_name": {"$eq": "retrieve_documents"}}


@patch("core.retriever.get_vectorstore_for_query")
def test_retrieve_documents_mmr_passes_symbol_filter(mock_get_vs):
    mock_vs = Mock()
    mock_vs.max_marginal_relevance_search.return_value = []
    mock_get_vs.return_value = mock_vs

    retrieve_documents_mmr("proj", "how does X work?", filter_symbol="retrieve_documents")

    kwargs = mock_vs.max_marginal_relevance_search.call_args.kwargs
    assert kwargs["filter"] == {"symbol_name": {"$eq": "retrieve_documents"}}


@patch("core.retriever.get_vectorstore_for_query")
def test_retrieve_documents_advanced_vector_mode_passes_symbol_filter_through(mock_get_vs):
    mock_vs = Mock()
    mock_vs.similarity_search.return_value = []
    mock_get_vs.return_value = mock_vs

    retrieve_documents_advanced("proj", "q", mode="vector", filter_symbol="retrieve_documents")

    kwargs = mock_vs.similarity_search.call_args.kwargs
    assert kwargs["filter"] == {"symbol_name": {"$eq": "retrieve_documents"}}


@patch("core.hybrid_retriever.hybrid_retrieve")
def test_retrieve_documents_advanced_hybrid_mode_passes_symbol_filter_as_where(mock_hybrid):
    mock_hybrid.return_value = [_doc("a.py")]

    retrieve_documents_advanced("proj", "q", mode="hybrid", filter_symbol="retrieve_documents")

    where_filter = mock_hybrid.call_args.kwargs.get("where_filter")
    assert where_filter == {"symbol_name": {"$eq": "retrieve_documents"}}