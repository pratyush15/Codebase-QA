
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from core.vectorstore import (
    add_documents_to_collection,
    _embed_batch_texts,
    _write_embedded_batch,
    get_collection_symbols,
)


def _docs(n, prefix="chunk"):
    return [
        Document(page_content=f"{prefix} {i}", metadata={"source": f"f{i}.py", "chunk_index": 0})
        for i in range(n)
    ]


def _fake_vectorstore(embed_side_effect=None):
    """
    A stand-in for the langchain_chroma.Chroma object returned by
    create_or_load_vectorstore: exposes ._embedding_function and
    ._collection the same way the real object does (the properties
    add_documents_to_collection relies on).
    """
    vs = Mock()
    vs._embedding_function = Mock()
    if embed_side_effect is not None:
        vs._embedding_function.embed_documents.side_effect = embed_side_effect
    else:
        vs._embedding_function.embed_documents.side_effect = (
            lambda texts: [[0.0, 0.0] for _ in texts]
        )
    vs._collection = Mock()
    return vs


# ---------- _embed_batch_texts / _write_embedded_batch (pure helpers) ----------

def test_embed_batch_texts_delegates_to_embedding_function():
    embedding_function = Mock()
    embedding_function.embed_documents.return_value = [[1.0, 2.0], [3.0, 4.0]]

    result = _embed_batch_texts(embedding_function, ["a", "b"])

    embedding_function.embed_documents.assert_called_once_with(["a", "b"])
    assert result == [[1.0, 2.0], [3.0, 4.0]]


def test_write_embedded_batch_upserts_ids_documents_metadata_embeddings():
    collection = Mock()
    batch = _docs(2)
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    _write_embedded_batch(collection, batch, embeddings)

    collection.upsert.assert_called_once()
    kwargs = collection.upsert.call_args.kwargs
    assert kwargs["embeddings"] == embeddings
    assert kwargs["documents"] == ["chunk 0", "chunk 1"]
    assert kwargs["metadatas"] == [batch[0].metadata, batch[1].metadata]
    assert len(kwargs["ids"]) == 2
    assert len(set(kwargs["ids"])) == 2  # unique per document


# ---------- add_documents_to_collection ----------

@patch("core.vectorstore.get_chroma_client")
@patch("core.vectorstore.create_or_load_vectorstore")
def test_empty_documents_still_creates_collection_and_returns_count(mock_create, mock_get_client):
    vs = _fake_vectorstore()
    mock_create.return_value = vs
    mock_get_client.return_value.get_collection.return_value.count.return_value = 0

    result = add_documents_to_collection("proj", [])

    mock_create.assert_called_once_with("proj")  # collection still created even with 0 docs
    vs._embedding_function.embed_documents.assert_not_called()
    vs._collection.upsert.assert_not_called()
    assert result == 0


@patch("core.vectorstore.get_chroma_client")
@patch("core.vectorstore.create_or_load_vectorstore")
def test_single_batch_uses_sequential_path_no_thread_pool(mock_create, mock_get_client):
    vs = _fake_vectorstore()
    mock_create.return_value = vs
    mock_get_client.return_value.get_collection.return_value.count.return_value = 3

    docs = _docs(3)
    result = add_documents_to_collection("proj", docs, batch_size=10, max_workers=4)

    vs._embedding_function.embed_documents.assert_called_once_with(["chunk 0", "chunk 1", "chunk 2"])
    vs._collection.upsert.assert_called_once()
    assert result == 3


@patch("core.vectorstore.get_chroma_client")
@patch("core.vectorstore.create_or_load_vectorstore")
def test_max_workers_1_forces_sequential_path_even_with_many_batches(mock_create, mock_get_client):
    vs = _fake_vectorstore()
    mock_create.return_value = vs
    mock_get_client.return_value.get_collection.return_value.count.return_value = 10

    docs = _docs(10)
    add_documents_to_collection("proj", docs, batch_size=2, max_workers=1)

    # 10 docs / batch_size 2 = 5 batches, each embedded and written separately
    assert vs._embedding_function.embed_documents.call_count == 5
    assert vs._collection.upsert.call_count == 5


@patch("core.vectorstore.get_chroma_client")
@patch("core.vectorstore.create_or_load_vectorstore")
def test_multiple_batches_all_embedded_and_written_via_thread_pool(mock_create, mock_get_client):
    vs = _fake_vectorstore()
    mock_create.return_value = vs
    mock_get_client.return_value.get_collection.return_value.count.return_value = 25

    docs = _docs(25)
    result = add_documents_to_collection("proj", docs, batch_size=5, max_workers=4)

    # 25 docs / batch_size 5 = 5 batches
    assert vs._embedding_function.embed_documents.call_count == 5
    assert vs._collection.upsert.call_count == 5
    # every document across every batch got embedded exactly once
    all_texts_embedded = [
        t for call in vs._embedding_function.embed_documents.call_args_list for t in call.args[0]
    ]
    assert sorted(all_texts_embedded) == sorted(d.page_content for d in docs)
    assert result == 25


@patch("core.vectorstore.get_chroma_client")
@patch("core.vectorstore.create_or_load_vectorstore")
def test_all_document_ids_are_unique_across_batches(mock_create, mock_get_client):
    vs = _fake_vectorstore()
    mock_create.return_value = vs
    mock_get_client.return_value.get_collection.return_value.count.return_value = 20

    docs = _docs(20)
    add_documents_to_collection("proj", docs, batch_size=4, max_workers=3)

    all_ids = [
        id_ for call in vs._collection.upsert.call_args_list for id_ in call.kwargs["ids"]
    ]
    assert len(all_ids) == 20
    assert len(set(all_ids)) == 20  # no collisions across concurrent batches


@patch("core.vectorstore.get_chroma_client")
@patch("core.vectorstore.create_or_load_vectorstore")
def test_embedding_failure_in_one_batch_propagates(mock_create, mock_get_client):
    def flaky_embed(texts):
        if "chunk 4" in texts:
            raise RuntimeError("ollama unreachable")
        return [[0.0, 0.0] for _ in texts]

    vs = _fake_vectorstore(embed_side_effect=flaky_embed)
    mock_create.return_value = vs

    docs = _docs(10)  # batch_size=2 -> "chunk 4" lands in one of the batches

    import pytest
    with pytest.raises(RuntimeError, match="ollama unreachable"):
        add_documents_to_collection("proj", docs, batch_size=2, max_workers=4)


@patch("core.vectorstore.get_chroma_client")
@patch("core.vectorstore.create_or_load_vectorstore")
def test_single_document_default_batch_and_worker_settings(mock_create, mock_get_client):
    # Sanity check against the real config-driven defaults (no explicit
    # batch_size/max_workers passed), since that's the code path actual
    # callers (core.indexing.sync_files_to_collection) hit.
    vs = _fake_vectorstore()
    mock_create.return_value = vs
    mock_get_client.return_value.get_collection.return_value.count.return_value = 1

    result = add_documents_to_collection("proj", _docs(1))

    vs._embedding_function.embed_documents.assert_called_once()
    assert result == 1


# ---------- chunk-level dedup ----------

def _dup_doc(source, chunk_content_hash, chunk_index=0, content="shared boilerplate"):
    return Document(
        page_content=f"# File: {source}\n---\n{content}",
        metadata={"source": source, "chunk_index": chunk_index, "chunk_content_hash": chunk_content_hash},
    )


@patch("core.vectorstore.get_chroma_client")
@patch("core.vectorstore.create_or_load_vectorstore")
def test_dedup_embeds_identical_chunk_content_only_once(mock_create, mock_get_client):
    vs = _fake_vectorstore()
    mock_create.return_value = vs
    mock_get_client.return_value.get_collection.return_value.count.return_value = 3

    # Three documents, but only two distinct chunk_content_hash values —
    # "vendor/a/helper.py" and "vendor/b/helper.py" share hash "h1".
    docs = [
        _dup_doc("vendor/a/helper.py", "h1"),
        _dup_doc("vendor/b/helper.py", "h1"),
        _dup_doc("app/main.py", "h2", content="unique code"),
    ]

    add_documents_to_collection("proj", docs, batch_size=10, max_workers=4)

    # only 2 unique texts embedded, not 3
    embedded_texts = vs._embedding_function.embed_documents.call_args.args[0]
    assert len(embedded_texts) == 2


@patch("core.vectorstore.get_chroma_client")
@patch("core.vectorstore.create_or_load_vectorstore")
def test_dedup_still_writes_every_original_document(mock_create, mock_get_client):
    vs = _fake_vectorstore()
    mock_create.return_value = vs
    mock_get_client.return_value.get_collection.return_value.count.return_value = 3

    docs = [
        _dup_doc("vendor/a/helper.py", "h1"),
        _dup_doc("vendor/b/helper.py", "h1"),
        _dup_doc("app/main.py", "h2", content="unique code"),
    ]

    add_documents_to_collection("proj", docs, batch_size=10, max_workers=4)

    # every original document still gets its own row written, with its
    # own distinct page_content/metadata (source path), even though only
    # one embedding was computed for the two duplicates
    upsert_kwargs = vs._collection.upsert.call_args.kwargs
    assert len(upsert_kwargs["documents"]) == 3
    assert len(upsert_kwargs["metadatas"]) == 3
    written_sources = {m["source"] for m in upsert_kwargs["metadatas"]}
    assert written_sources == {"vendor/a/helper.py", "vendor/b/helper.py", "app/main.py"}


@patch("core.vectorstore.get_chroma_client")
@patch("core.vectorstore.create_or_load_vectorstore")
def test_dedup_duplicate_documents_share_the_same_embedding_vector(mock_create, mock_get_client):
    vs = _fake_vectorstore()
    mock_create.return_value = vs
    mock_get_client.return_value.get_collection.return_value.count.return_value = 2

    docs = [_dup_doc("vendor/a/helper.py", "h1"), _dup_doc("vendor/b/helper.py", "h1")]
    add_documents_to_collection("proj", docs, batch_size=10, max_workers=4)

    upsert_kwargs = vs._collection.upsert.call_args.kwargs
    embeddings = upsert_kwargs["embeddings"]
    assert embeddings[0] == embeddings[1]


@patch("core.vectorstore.get_chroma_client")
@patch("core.vectorstore.create_or_load_vectorstore")
def test_dedup_ids_stay_unique_even_for_duplicate_content(mock_create, mock_get_client):
    vs = _fake_vectorstore()
    mock_create.return_value = vs
    mock_get_client.return_value.get_collection.return_value.count.return_value = 2

    docs = [_dup_doc("vendor/a/helper.py", "h1"), _dup_doc("vendor/b/helper.py", "h1")]
    add_documents_to_collection("proj", docs, batch_size=10, max_workers=4)

    ids = vs._collection.upsert.call_args.kwargs["ids"]
    assert len(ids) == 2
    assert ids[0] != ids[1]  # distinct Chroma rows despite sharing an embedding


@patch("core.vectorstore.get_chroma_client")
@patch("core.vectorstore.create_or_load_vectorstore")
def test_documents_without_chunk_content_hash_are_never_deduped(mock_create, mock_get_client):
    # Documents missing the chunk_content_hash metadata key (e.g. from an
    # older index, or a caller that didn't go through ingest_files) must
    # never be merged together just because they lack a hash.
    vs = _fake_vectorstore()
    mock_create.return_value = vs
    mock_get_client.return_value.get_collection.return_value.count.return_value = 2

    docs = _docs(2)  # no chunk_content_hash metadata set
    add_documents_to_collection("proj", docs, batch_size=10, max_workers=4)

    embedded_texts = vs._embedding_function.embed_documents.call_args.args[0]
    assert len(embedded_texts) == 2  # both still embedded, nothing merged


@patch("core.vectorstore.get_chroma_client")
@patch("core.vectorstore.create_or_load_vectorstore")
def test_dedup_spans_across_batches_not_just_within_one(mock_create, mock_get_client):
    vs = _fake_vectorstore()
    mock_create.return_value = vs
    mock_get_client.return_value.get_collection.return_value.count.return_value = 4

    # 4 documents sharing 1 hash, batch_size=1 forces them into 4 separate
    # batches — dedup must still catch it across batch boundaries.
    docs = [_dup_doc(f"vendor/copy_{i}/helper.py", "h1") for i in range(4)]
    add_documents_to_collection("proj", docs, batch_size=1, max_workers=4)

    assert vs._embedding_function.embed_documents.call_count == 1
    assert vs._collection.upsert.call_count == 4  # still written as 4 separate rows


@patch("core.vectorstore.get_chroma_client")
@patch("core.vectorstore.create_or_load_vectorstore")
def test_no_duplicates_behaves_identically_to_before(mock_create, mock_get_client):
    vs = _fake_vectorstore()
    mock_create.return_value = vs
    mock_get_client.return_value.get_collection.return_value.count.return_value = 3

    docs = [_dup_doc(f"file_{i}.py", f"h{i}", content=f"code {i}") for i in range(3)]
    add_documents_to_collection("proj", docs, batch_size=10, max_workers=4)

    embedded_texts = vs._embedding_function.embed_documents.call_args.args[0]
    assert len(embedded_texts) == 3


# ---------- get_collection_symbols ----------

@patch("core.vectorstore.get_chroma_client")
def test_get_collection_symbols_returns_sorted_unique_symbols(mock_get_client):
    mock_col = Mock()
    mock_col.get.return_value = {
        "metadatas": [
            {"symbol_name": "retrieve_documents"},
            {"symbol_name": "ingest_files"},
            {"symbol_name": "retrieve_documents"},  # duplicate, from another chunk
        ]
    }
    mock_get_client.return_value.get_collection.return_value = mock_col

    result = get_collection_symbols("proj")

    assert result == ["ingest_files", "retrieve_documents"]


@patch("core.vectorstore.get_chroma_client")
def test_get_collection_symbols_skips_empty_symbol_names(mock_get_client):
    mock_col = Mock()
    mock_col.get.return_value = {
        "metadatas": [
            {"symbol_name": "foo"},
            {"symbol_name": ""},       # heuristic-chunked / non-definition chunk
            {},                        # missing key entirely
        ]
    }
    mock_get_client.return_value.get_collection.return_value = mock_col

    result = get_collection_symbols("proj")

    assert result == ["foo"]


@patch("core.vectorstore.get_chroma_client")
def test_get_collection_symbols_returns_empty_list_on_failure(mock_get_client):
    mock_get_client.return_value.get_collection.side_effect = RuntimeError("collection not found")

    assert get_collection_symbols("proj") == []


@patch("core.vectorstore.get_chroma_client")
def test_get_collection_symbols_empty_collection_returns_empty_list(mock_get_client):
    mock_col = Mock()
    mock_col.get.return_value = {"metadatas": []}
    mock_get_client.return_value.get_collection.return_value = mock_col

    assert get_collection_symbols("proj") == []