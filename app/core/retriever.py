
import logging
from typing import List, Optional, Dict
from langchain_core.documents import Document

from config import (
    MAX_RETRIEVAL_DOCS,
    RETRIEVAL_MODE,
    RETRIEVAL_FETCH_K,
    MMR_LAMBDA,
    RRF_K,
    RERANKER_MODEL,
)
from core.vectorstore import get_vectorstore_for_query

logger = logging.getLogger(__name__)


def retrieve_documents(
    collection_name: str,
    query: str,
    k: int = MAX_RETRIEVAL_DOCS,
    filter_language: Optional[str] = None,
    filter_source: Optional[str] = None,
    filter_symbol: Optional[str] = None,
) -> List[Document]:
    """
    Plain top-k vector similarity search. Kept as-is (original behavior)
    for callers that want the simplest, fastest path. See
    retrieve_documents_advanced for MMR / hybrid / reranked retrieval.

    filter_symbol: exact match against a chunk's symbol_name metadata
    (the function/class/method it defines — see core.ast_chunking), for
    "show me the `retrieve_documents` function" style scoping. Only
    chunks produced by AST chunking carry a symbol_name at all; chunks
    from the heuristic splitter never match a symbol filter.
    """
    vectorstore = get_vectorstore_for_query(collection_name)
    if vectorstore is None:
        return []

    where_filter = _build_filter(filter_language, filter_source, filter_symbol)

    try:
        if where_filter:
            return vectorstore.similarity_search(query, k=k, filter=where_filter)
        return vectorstore.similarity_search(query, k=k)
    except Exception:
        logger.exception(
            "Similarity search failed for collection '%s' (query=%r, filter=%r)",
            collection_name, query, where_filter,
        )
        return []


def retrieve_documents_mmr(
    collection_name: str,
    query: str,
    k: int = MAX_RETRIEVAL_DOCS,
    fetch_k: int = RETRIEVAL_FETCH_K,
    lambda_mult: float = MMR_LAMBDA,
    filter_language: Optional[str] = None,
    filter_source: Optional[str] = None,
    filter_symbol: Optional[str] = None,
) -> List[Document]:
    """
    Vector search with maximal marginal relevance: still ranks by
    similarity to the query, but penalizes candidates that are too similar
    to ones already selected — so top-k isn't 6 near-duplicate chunks from
    the same function when the collection has repetitive code.
    """
    vectorstore = get_vectorstore_for_query(collection_name)
    if vectorstore is None:
        return []

    where_filter = _build_filter(filter_language, filter_source, filter_symbol)

    try:
        kwargs = {"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
        if where_filter:
            kwargs["filter"] = where_filter
        return vectorstore.max_marginal_relevance_search(query, **kwargs)
    except Exception:
        logger.exception(
            "MMR search failed for collection '%s' (query=%r, filter=%r)",
            collection_name, query, where_filter,
        )
        return []


def retrieve_documents_advanced(
    collection_name: str,
    query: str,
    k: int = MAX_RETRIEVAL_DOCS,
    mode: str = RETRIEVAL_MODE,
    filter_language: Optional[str] = None,
    filter_source: Optional[str] = None,
    filter_symbol: Optional[str] = None,
    extra_queries: Optional[List[str]] = None,
) -> List[Document]:
    """
    Unified retrieval entry point. mode selects the retrieval strategy:

      "vector"        — retrieve_documents() (plain top-k cosine similarity)
      "mmr"           — retrieve_documents_mmr() (diversity-aware top-k)
      "hybrid"        — BM25 + vector fused via reciprocal rank fusion
      "hybrid_rerank" — hybrid, then re-ordered by a cross-encoder

    filter_symbol: exact match against symbol_name metadata — see
    retrieve_documents's docstring. Applies uniformly across every mode,
    same as filter_language/filter_source.

    extra_queries: additional phrasings of `query` (e.g. LLM-generated
    paraphrases from core.query_rewrite.expand_query) to fuse into the
    ranking via RAG-Fusion-style multi-query retrieval. Only applies to
    the hybrid/hybrid_rerank paths — ignored for "vector"/"mmr", since
    those aren't RRF-fusion-based to begin with. None/[] behaves exactly
    like passing nothing (no behavior change for existing callers).

    Falls back one step on failure/unavailability rather than erroring the
    whole query: hybrid_rerank -> hybrid if the reranker isn't available,
    hybrid -> vector if BM25/vectorstore setup fails.
    """
    where_filter = _build_filter(filter_language, filter_source, filter_symbol)

    if mode == "vector":
        return retrieve_documents(collection_name, query, k, filter_language, filter_source, filter_symbol)

    if mode == "mmr":
        return retrieve_documents_mmr(
            collection_name, query, k,
            filter_language=filter_language, filter_source=filter_source, filter_symbol=filter_symbol,
        )

    if mode in ("hybrid", "hybrid_rerank"):
        from core.hybrid_retriever import hybrid_retrieve, hybrid_retrieve_multi_query

        try:
            if extra_queries:
                fused = hybrid_retrieve_multi_query(
                    collection_name, [query] + list(extra_queries), k=RETRIEVAL_FETCH_K,
                    fetch_k=RETRIEVAL_FETCH_K, where_filter=where_filter,
                )
            else:
                fused = hybrid_retrieve(
                    collection_name, query, k=RETRIEVAL_FETCH_K,
                    fetch_k=RETRIEVAL_FETCH_K, where_filter=where_filter,
                )
        except Exception:
            logger.exception("Hybrid retrieval failed for '%s' — falling back to plain vector search", collection_name)
            return retrieve_documents(collection_name, query, k, filter_language, filter_source, filter_symbol)

        if mode == "hybrid":
            return fused[:k]

        # hybrid_rerank
        from core.reranker import rerank, is_reranking_available

        if not is_reranking_available():
            logger.info("Reranker unavailable — returning hybrid fusion result unranked.")
            return fused[:k]
        return rerank(query, fused, top_k=k, model_name=RERANKER_MODEL)

    logger.warning("Unknown retrieval mode '%s' — falling back to plain vector search", mode)
    return retrieve_documents(collection_name, query, k, filter_language, filter_source, filter_symbol)


def _build_filter(
    filter_language: Optional[str],
    filter_source: Optional[str],
    filter_symbol: Optional[str] = None,
) -> Optional[Dict]:
    conditions = []
    if filter_language:
        conditions.append({"language": {"$eq": filter_language}})
    if filter_source:
        conditions.append({"source": {"$eq": filter_source}})
    if filter_symbol:
        conditions.append({"symbol_name": {"$eq": filter_symbol}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def assemble_context(docs: List[Document]) -> str:
    seen = set()
    parts = []

    for doc in docs:
        meta = doc.metadata
        key = (meta.get("source", ""), meta.get("chunk_index", 0))
        if key in seen:
            continue
        seen.add(key)
        parts.append(doc.page_content)

    return "\n\n---\n\n".join(parts)


def get_source_summary(docs: List[Document]) -> List[Dict]:
    seen = set()
    sources = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        if source in seen:
            continue
        seen.add(source)
        sources.append({
            "source": source,
            "language": doc.metadata.get("language", ""),
            "role": doc.metadata.get("role", ""),
            "chunk_index": doc.metadata.get("chunk_index", 0),
            "total_chunks": doc.metadata.get("total_chunks", 1),
            "symbol_name": doc.metadata.get("symbol_name", ""),
            "symbol_type": doc.metadata.get("symbol_type", ""),
            "parent_symbol": doc.metadata.get("parent_symbol", ""),
        })

    return sources