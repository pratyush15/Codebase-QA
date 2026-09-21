
from langchain_core.documents import Document

from core.hybrid_retriever import tokenize, bm25_search, reciprocal_rank_fusion


def _doc(source, chunk_index, content):
    return Document(page_content=content, metadata={"source": source, "chunk_index": chunk_index})


# ---------- tokenize ----------

def test_tokenize_splits_snake_case_into_subwords_and_keeps_whole_identifier():
    tokens = tokenize("get_user_by_id")
    assert "get_user_by_id" in tokens
    assert "get" in tokens
    assert "user" in tokens
    assert "by" in tokens
    assert "id" in tokens


def test_tokenize_is_case_insensitive():
    assert tokenize("GetUser") == tokenize("getuser")


def test_tokenize_ignores_punctuation():
    tokens = tokenize("db.query(id);")
    assert "db" in tokens
    assert "query" in tokens
    assert "id" in tokens
    assert ";" not in tokens
    assert "." not in tokens


# ---------- bm25_search ----------

def test_bm25_search_ranks_matching_doc_above_unrelated_ones():
    docs = [
        _doc("a.py", 0, "def get_user_by_id(id): return db.query(id)"),
        _doc("b.py", 0, "def list_products(): return db.all()"),
        _doc("c.py", 0, "def delete_order(order_id): db.remove(order_id)"),
        _doc("d.py", 0, "def compute_tax(amount): return amount * 0.1"),
        _doc("e.py", 0, "class UserRepository:\n    def find_user(self, id): pass"),
    ]
    results = bm25_search(docs, "find user by id", k=5)
    result_sources = [d.metadata["source"] for d in results]

    assert result_sources[0] in ("e.py", "a.py")  # most on-topic matches first
    assert "d.py" not in result_sources or result_sources.index("d.py") > 1  # unrelated doc ranks low if present at all


def test_bm25_search_returns_empty_list_for_empty_corpus():
    assert bm25_search([], "anything", k=5) == []


def test_bm25_search_respects_k():
    docs = [_doc(f"f{i}.py", 0, f"def handler_{i}(): return {i}") for i in range(10)]
    results = bm25_search(docs, "handler", k=3)
    assert len(results) <= 3


# ---------- reciprocal_rank_fusion ----------

def test_rrf_ranks_doc_found_by_both_retrievers_first():
    doc_a = _doc("a.py", 0, "content a")
    doc_b = _doc("b.py", 0, "content b")
    doc_c = _doc("c.py", 0, "content c")

    vector_ranking = [doc_a, doc_b, doc_c]
    bm25_ranking = [doc_b, doc_a]  # doc_b ranks higher here

    fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking])
    fused_sources = [d.metadata["source"] for d in fused]

    # doc_a (rank 1+2) and doc_b (rank 2+1) both appear in both rankings and
    # should outrank doc_c, which only one retriever found.
    assert fused_sources.index("c.py") > fused_sources.index("a.py")
    assert fused_sources.index("c.py") > fused_sources.index("b.py")


def test_rrf_deduplicates_same_doc_across_rankings():
    doc_a = _doc("a.py", 0, "content a")
    fused = reciprocal_rank_fusion([[doc_a], [doc_a], [doc_a]])
    assert len(fused) == 1


def test_rrf_handles_empty_rankings():
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_single_ranking_preserves_order():
    doc_a = _doc("a.py", 0, "a")
    doc_b = _doc("b.py", 0, "b")
    fused = reciprocal_rank_fusion([[doc_a, doc_b]])
    assert [d.metadata["source"] for d in fused] == ["a.py", "b.py"]

# ---------- build_bm25_index / bm25_search_with_index ----------

from unittest.mock import Mock, patch
from core.hybrid_retriever import build_bm25_index, bm25_search_with_index, hybrid_retrieve_multi_query


def test_build_bm25_index_returns_none_for_empty_corpus():
    assert build_bm25_index([]) is None


def test_bm25_search_with_index_matches_bm25_search_results():
    from core.hybrid_retriever import bm25_search

    docs = [
        _doc("a.py", 0, "def get_user_by_id(id): return db.query(id)"),
        _doc("b.py", 0, "def list_products(): return db.all()"),
    ]
    index = build_bm25_index(docs)
    via_index = bm25_search_with_index(index, docs, "find user by id", k=5)
    via_one_shot = bm25_search(docs, "find user by id", k=5)

    assert [d.metadata["source"] for d in via_index] == [d.metadata["source"] for d in via_one_shot]


def test_bm25_search_with_index_returns_empty_for_none_index():
    docs = [_doc("a.py", 0, "def foo(): pass")]
    assert bm25_search_with_index(None, docs, "foo", k=5) == []


# ---------- hybrid_retrieve_multi_query ----------

def test_multi_query_returns_empty_for_no_queries():
    assert hybrid_retrieve_multi_query("proj", []) == []


@patch("core.hybrid_retriever.get_all_documents")
@patch("core.hybrid_retriever.get_vectorstore_for_query")
def test_multi_query_returns_empty_when_no_vectorstore(mock_get_vs, mock_get_all):
    mock_get_vs.return_value = None
    assert hybrid_retrieve_multi_query("proj", ["q1", "q2"]) == []
    mock_get_all.assert_not_called()


@patch("core.hybrid_retriever.get_all_documents")
@patch("core.hybrid_retriever.get_vectorstore_for_query")
def test_multi_query_runs_vector_and_bm25_search_per_query_variant(mock_get_vs, mock_get_all):
    doc_a = _doc("a.py", 0, "def get_user_by_id(id): return db.query(id)")
    doc_b = _doc("b.py", 0, "def authenticate_session(token): validate(token)")

    mock_vectorstore = Mock()
    mock_vectorstore.similarity_search.return_value = [doc_a]
    mock_get_vs.return_value = mock_vectorstore
    mock_get_all.return_value = [doc_a, doc_b]

    queries = ["find user by id", "authenticate a session"]
    hybrid_retrieve_multi_query("proj", queries, k=5, fetch_k=5)

    # one vector search per query variant
    assert mock_vectorstore.similarity_search.call_count == 2
    called_queries = [c.args[0] for c in mock_vectorstore.similarity_search.call_args_list]
    assert called_queries == queries

    # corpus fetched once, not once per query variant
    mock_get_all.assert_called_once()


@patch("core.hybrid_retriever.get_all_documents")
@patch("core.hybrid_retriever.get_vectorstore_for_query")
def test_multi_query_a_doc_found_by_multiple_variants_ranks_higher(mock_get_vs, mock_get_all):
    doc_a = _doc("a.py", 0, "def authenticate_session(token): validate(token)")
    doc_b = _doc("b.py", 0, "def unrelated_helper(): pass")

    mock_vectorstore = Mock()
    # doc_a shows up for both query phrasings; doc_b only for one.
    mock_vectorstore.similarity_search.side_effect = [
        [doc_a],           # "log a user in"
        [doc_a, doc_b],    # "authenticate a session"
    ]
    mock_get_vs.return_value = mock_vectorstore
    mock_get_all.return_value = []  # empty corpus -> bm25 contributes nothing, isolates vector fusion

    fused = hybrid_retrieve_multi_query(
        "proj", ["log a user in", "authenticate a session"], k=5, fetch_k=5,
    )
    fused_sources = [d.metadata["source"] for d in fused]

    assert fused_sources[0] == "a.py"
    assert "b.py" in fused_sources


@patch("core.hybrid_retriever.get_all_documents")
@patch("core.hybrid_retriever.get_vectorstore_for_query")
def test_multi_query_single_query_behaves_like_hybrid_retrieve(mock_get_vs, mock_get_all):
    from core.hybrid_retriever import hybrid_retrieve

    doc_a = _doc("a.py", 0, "def get_user_by_id(id): return db.query(id)")
    doc_b = _doc("b.py", 0, "def list_products(): return db.all()")

    mock_vectorstore = Mock()
    mock_vectorstore.similarity_search.return_value = [doc_a]
    mock_get_vs.return_value = mock_vectorstore
    mock_get_all.return_value = [doc_a, doc_b]

    single = hybrid_retrieve("proj", "find user by id", k=5, fetch_k=5)
    multi = hybrid_retrieve_multi_query("proj", ["find user by id"], k=5, fetch_k=5)

    assert [d.metadata["source"] for d in single] == [d.metadata["source"] for d in multi]


@patch("core.hybrid_retriever.get_all_documents")
@patch("core.hybrid_retriever.get_vectorstore_for_query")
def test_multi_query_continues_on_vector_search_failure_for_one_variant(mock_get_vs, mock_get_all):
    doc_a = _doc("a.py", 0, "def foo(): pass")

    mock_vectorstore = Mock()
    mock_vectorstore.similarity_search.side_effect = [RuntimeError("boom"), [doc_a]]
    mock_get_vs.return_value = mock_vectorstore
    mock_get_all.return_value = [doc_a]

    # should not raise despite the first variant's vector search failing
    fused = hybrid_retrieve_multi_query("proj", ["bad query", "good query"], k=5, fetch_k=5)
    assert any(d.metadata["source"] == "a.py" for d in fused)