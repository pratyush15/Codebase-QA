
import os
import re
import uuid
import logging
import concurrent.futures
from typing import List, Optional, Dict

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config import (
    CHROMA_DB_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    MAX_EMBEDDING_WORKERS,
)

logger = logging.getLogger(__name__)


def get_embedding_function():
    return OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_EMBEDDING_MODEL,
    )


def get_chroma_client() -> chromadb.PersistentClient:
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def sanitize_collection_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9\-]", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    name = name[:63]
    if len(name) < 3:
        name = name + "-qa"
    return name.lower()


def list_collections(exclude_prefixes: tuple = ("eval-",)) -> List[Dict]:
    """
    List Chroma collections in the shared store, for the Streamlit project
    selector.

    exclude_prefixes filters out internal/benchmarking collections (e.g.
    the eval harness's "eval-codebase-qa-self") that live in the same
    Chroma store but aren't a real project a user uploaded — they'd
    otherwise clutter the "Active project" dropdown. Pass exclude_prefixes=()
    to see everything, e.g. for an admin/debug view.
    """
    client = get_chroma_client()
    result = []
    for col in client.list_collections():
        if exclude_prefixes and col.name.startswith(exclude_prefixes):
            continue
        try:
            count = client.get_collection(col.name).count()
        except Exception:
            logger.exception("Failed to get chunk count for collection '%s'", col.name)
            count = 0
        result.append({"name": col.name, "chunk_count": count})
    return sorted(result, key=lambda x: x["name"])


def collection_exists(collection_name: str) -> bool:
    client = get_chroma_client()
    return collection_name in [c.name for c in client.list_collections()]


def create_or_load_vectorstore(collection_name: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_function(),
        persist_directory=CHROMA_DB_PATH,
    )


def _embed_batch_texts(embedding_function, texts: List[str]) -> List[List[float]]:
    """Runs on a worker thread: one HTTP round trip to Ollama's embed endpoint, no shared state."""
    return embedding_function.embed_documents(texts)


def _write_embedded_batch(collection: chromadb.Collection, batch: List[Document], embeddings: List) -> None:
    """
    Runs on the calling (main) thread only — chromadb's client isn't
    guaranteed safe for concurrent writers, so every write goes through
    here, one batch at a time, never overlapping with another write.
    """
    collection.upsert(
        ids=[str(uuid.uuid4()) for _ in batch],
        embeddings=embeddings,
        documents=[d.page_content for d in batch],
        metadatas=[d.metadata for d in batch],
    )


def _dedup_key(doc: Document) -> str:
    """
    Groups documents whose *chunk content* is identical (vendored copies,
    generated boilerplate, a shared license header) so they can share one
    embedding call instead of paying for the same embedding N times.

    Keyed off metadata["chunk_content_hash"] (see
    core.ingestion.hash_chunk_content — hashed on the raw chunk, before
    the per-file "# File: ..." header is prepended, so two files with
    identical code but different paths are still recognized as the same
    content). A document missing that metadata key falls back to its
    object identity, i.e. it's simply never deduped against anything —
    safer than accidentally merging unrelated content under a shared
    empty-string key.
    """
    key = doc.metadata.get("chunk_content_hash")
    return key if key else f"__no_hash__:{id(doc)}"


def _embed_unique_batches(
    batches: List[List[Document]],
    embedding_function,
    max_workers: int,
) -> Dict[str, List[float]]:
    """
    Embeds already-deduplicated batches (one representative Document per
    unique chunk) and returns {dedup_key: embedding}. Batches run
    concurrently via a thread pool when there's more than one and
    max_workers > 1; otherwise falls back to a plain sequential loop, same
    as add_documents_to_collection's single-batch fast path.
    """
    key_to_embedding: Dict[str, List[float]] = {}

    def _record(batch: List[Document], embeddings: List[List[float]]) -> None:
        for doc, embedding in zip(batch, embeddings):
            key_to_embedding[_dedup_key(doc)] = embedding

    if max_workers <= 1 or len(batches) <= 1:
        for batch in batches:
            _record(batch, _embed_batch_texts(embedding_function, [d.page_content for d in batch]))
        return key_to_embedding

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(_embed_batch_texts, embedding_function, [d.page_content for d in batch]): batch
            for batch in batches
        }
        try:
            for future in concurrent.futures.as_completed(future_to_batch):
                batch = future_to_batch[future]
                embeddings = future.result()  # re-raises here if this batch's embedding call failed
                _record(batch, embeddings)
        except Exception:
            for pending in future_to_batch:
                pending.cancel()
            raise

    return key_to_embedding


def add_documents_to_collection(
    collection_name: str,
    documents: list,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    max_workers: int = MAX_EMBEDDING_WORKERS,
) -> int:
    """
    Embeds `documents` and writes them into collection_name, returning the
    collection's total chunk count afterward.

    Two optimizations over a naive "embed everything, one call per batch":

    1. Chunk-level dedup. Identical chunk content shows up in more than one
       file surprisingly often — vendored copies of a dependency, generated
       code, a license header pasted into every source file. Embedding is
       by far the most expensive part of indexing, so each unique chunk
       (by core.ingestion.hash_chunk_content) is embedded exactly once and
       its vector is reused for every document that shares that content.
       Every document is still written to Chroma individually — dedup only
       skips redundant *embedding calls*, not storage, so per-file
       metadata, provenance, and filter_source lookups are unaffected. This
       is scoped to the documents passed into a single call: it doesn't
       check whether an identical chunk already has an embedding stored
       from a previous indexing run elsewhere in the collection.

    2. Concurrent embedding. Each unique-chunk batch is an independent
       network-bound HTTP round trip to Ollama with no shared state, so up
       to `max_workers` batches are embedded concurrently rather than one
       after another. Writing the results into Chroma is deliberately NOT
       parallelized — chromadb's client isn't guaranteed safe for
       concurrent writers — so every write stays on the calling thread,
       one batch at a time.

    If any batch's embedding call fails, still-pending batches are
    cancelled (already in-flight ones are allowed to finish, since a
    thread mid-HTTP-call can't be aborted) and the exception propagates —
    the whole call fails loudly rather than silently indexing a partial
    result.
    """
    vectorstore = create_or_load_vectorstore(collection_name)

    if not documents:
        return get_chroma_client().get_collection(collection_name).count()

    embedding_function = vectorstore._embedding_function
    collection = vectorstore._collection

    seen_keys = set()
    unique_docs = []
    for doc in documents:
        key = _dedup_key(doc)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_docs.append(doc)

    duplicate_count = len(documents) - len(unique_docs)
    if duplicate_count:
        logger.info(
            "Chunk dedup for collection '%s': %d/%d chunk(s) share content with an "
            "earlier chunk in this batch — embedding only %d unique chunk(s).",
            collection_name, duplicate_count, len(documents), len(unique_docs),
        )

    embed_batches = [unique_docs[i:i + batch_size] for i in range(0, len(unique_docs), batch_size)]
    key_to_embedding = _embed_unique_batches(embed_batches, embedding_function, max_workers)

    for i in range(0, len(documents), batch_size):
        write_batch = documents[i:i + batch_size]
        embeddings = [key_to_embedding[_dedup_key(doc)] for doc in write_batch]
        _write_embedded_batch(collection, write_batch, embeddings)

    return get_chroma_client().get_collection(collection_name).count()


def delete_collection(collection_name: str) -> bool:
    try:
        get_chroma_client().delete_collection(collection_name)
        return True
    except Exception:
        logger.exception("Failed to delete collection '%s'", collection_name)
        return False


def get_collection_sources(collection_name: str) -> List[str]:
    try:
        col = get_chroma_client().get_collection(collection_name)
        result = col.get(include=["metadatas"])
        sources = {m["source"] for m in result.get("metadatas", []) if m and "source" in m}
        return sorted(sources)
    except Exception:
        logger.exception("Failed to fetch sources for collection '%s'", collection_name)
        return []


def get_collection_symbols(collection_name: str) -> List[str]:
    """
    Sorted list of every distinct, non-empty symbol_name currently indexed
    in this collection (function/class/method names extracted by AST
    chunking — see core.ast_chunking). Powers the "scope to symbol" filter
    in the Streamlit sidebar, the same way get_collection_sources powers
    "scope to file". Files chunked by the heuristic splitter (languages
    without a tree-sitter grammar wired up) contribute no symbols here,
    since that path can't identify them.
    """
    try:
        col = get_chroma_client().get_collection(collection_name)
        result = col.get(include=["metadatas"])
        symbols = {m["symbol_name"] for m in result.get("metadatas", []) if m and m.get("symbol_name")}
        return sorted(symbols)
    except Exception:
        logger.exception("Failed to fetch symbols for collection '%s'", collection_name)
        return []


def get_collection_file_hashes(collection_name: str) -> Dict[str, str]:
    """
    Map of source path -> content_hash for every file currently indexed in
    this collection. All chunks of a given file share the same content_hash
    (it's a per-file hash, not a per-chunk one), so the first chunk seen for
    a source is representative of the whole file.

    Used by core.indexing.sync_files_to_collection to skip re-embedding
    files that haven't changed since the last index run.
    """
    if not collection_exists(collection_name):
        # Expected right after a fresh delete_collection() call (e.g. the
        # eval harness's --reindex flag) — there's nothing to hash yet, so
        # every file is correctly treated as new. Not an error condition;
        # logging it at ERROR with a full traceback was misleading.
        logger.info(
            "Collection '%s' does not exist yet — treating all files as new.",
            collection_name,
        )
        return {}

    try:
        col = get_chroma_client().get_collection(collection_name)
        result = col.get(include=["metadatas"])
    except Exception:
        logger.exception("Failed to fetch file hashes for collection '%s'", collection_name)
        return {}

    hashes: Dict[str, str] = {}
    for meta in result.get("metadatas", []):
        if not meta:
            continue
        source = meta.get("source")
        content_hash = meta.get("content_hash")
        if source and content_hash and source not in hashes:
            hashes[source] = content_hash
    return hashes


def delete_documents_by_source(collection_name: str, source: str) -> None:
    """Delete all chunks belonging to a single file, ahead of re-adding updated chunks for it."""
    try:
        col = get_chroma_client().get_collection(collection_name)
        col.delete(where={"source": {"$eq": source}})
    except Exception:
        logger.exception(
            "Failed to delete existing chunks for source '%s' in collection '%s'",
            source, collection_name,
        )


def delete_documents_by_sources(collection_name: str, sources: List[str]) -> None:
    """
    Delete all chunks belonging to any of the given files, in one Chroma
    call. Used for stale-file cleanup during a full sync (see
    core.indexing.sync_files_to_collection), where a re-upload can name
    many files that were removed from the project since the last index.
    """
    if not sources:
        return
    try:
        col = get_chroma_client().get_collection(collection_name)
        if len(sources) == 1:
            col.delete(where={"source": {"$eq": sources[0]}})
        else:
            col.delete(where={"source": {"$in": sources}})
    except Exception:
        logger.exception(
            "Failed to delete chunks for %d stale source(s) in collection '%s'",
            len(sources), collection_name,
        )


def get_all_documents(collection_name: str, where_filter: Optional[Dict] = None) -> List["Document"]:
    """
    Fetch every chunk in a collection as LangChain Documents (page_content +
    metadata), reconstructed from Chroma's raw get() response. Used by BM25
    keyword search, which needs the full corpus rather than a vector top-k.
    """
    from langchain_core.documents import Document

    try:
        col = get_chroma_client().get_collection(collection_name)
        kwargs = {"include": ["documents", "metadatas"]}
        if where_filter:
            kwargs["where"] = where_filter
        result = col.get(**kwargs)
    except Exception:
        logger.exception("Failed to fetch all documents for collection '%s'", collection_name)
        return []

    docs = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    return [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(docs, metadatas)
    ]


def get_vectorstore_for_query(collection_name: str) -> Optional[Chroma]:
    if not collection_exists(collection_name):
        return None
    return create_or_load_vectorstore(collection_name)