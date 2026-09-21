
"""
Hybrid retrieval: BM25 (keyword) + vector similarity, merged with
Reciprocal Rank Fusion (RRF).

Why: pure vector similarity search misses exact-term lookups — searching
for a specific function/variable/error-string name often scores lower on
embedding similarity than a semantically related but differently-named
chunk. BM25 catches exactly the cases embeddings miss, and vice versa, so
fusing both rankings covers more ground than either alone.

The BM25 index is rebuilt from scratch on every query, from whatever's
currently in the collection. That's fine at the scale of one indexed
codebase (hundreds to low thousands of chunks) — for a much larger corpus,
this is the place a persistent/cached BM25 index would go instead.

hybrid_retrieve_multi_query() extends this with query expansion / RAG
Fusion: a user's question rarely shares vocabulary with the identifiers in
the code that actually answers it (e.g. "how do I log a user in" vs. a
codebase that says "authenticate_session"). Given a handful of LLM-
generated paraphrases of the question (see core.query_rewrite.expand_query),
it runs vector + BM25 search for *each* phrasing and fuses every one of
those rankings together — a chunk that only one phrasing's wording happens
to match still gets a fair shot at surfacing, and a chunk multiple
phrasings agree on rises further.
"""
import re
import logging
from typing import List, Dict, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from core.vectorstore import get_all_documents, get_vectorstore_for_query

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def tokenize(text: str) -> List[str]:
    """
    Code-aware tokenizer for BM25: in addition to each raw identifier,
    also emits its snake_case parts, so a query like 'get user' can match
    a chunk containing 'get_user' / 'get_user_by_id'.
    """
    tokens: List[str] = []
    for match in _TOKEN_RE.findall(text.lower()):
        tokens.append(match)
        parts = match.split("_")
        if len(parts) > 1:
            tokens.extend(p for p in parts if p)
    return tokens


def _doc_key(doc: Document) -> Tuple[str, int]:
    return (doc.metadata.get("source", ""), doc.metadata.get("chunk_index", 0))


def build_bm25_index(all_docs: List[Document]) -> Optional[BM25Okapi]:
    """
    Builds a BM25Okapi index once for a corpus, so callers that need to
    query the same corpus multiple times (e.g. once per paraphrase in
    multi-query retrieval) don't pay the tokenize-and-index cost N times
    over. Returns None for an empty corpus.
    """
    if not all_docs:
        return None
    tokenized_corpus = [tokenize(d.page_content) for d in all_docs]
    return BM25Okapi(tokenized_corpus)


def bm25_search_with_index(
    bm25_index: Optional[BM25Okapi], all_docs: List[Document], query: str, k: int,
) -> List[Document]:
    """Query a pre-built BM25 index. Returns the top k docs with score > 0."""
    if bm25_index is None or not all_docs:
        return []

    scores = bm25_index.get_scores(tokenize(query))
    ranked = sorted(zip(all_docs, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, score in ranked[:k] if score > 0]


def bm25_search(all_docs: List[Document], query: str, k: int) -> List[Document]:
    """Single-shot convenience wrapper: builds the index, then searches once."""
    return bm25_search_with_index(build_bm25_index(all_docs), all_docs, query, k)


def reciprocal_rank_fusion(rankings: List[List[Document]], rrf_k: int = 60) -> List[Document]:
    """
    Merge multiple ranked lists into one fused ranking. Each document's
    fused score is sum(1 / (rrf_k + rank)) across every input ranking it
    appears in (rank is 1-based within that ranking) — documents found by
    multiple retrievers naturally float to the top, without needing the
    raw scores from each retriever to be on comparable scales.
    """
    scores: Dict[Tuple[str, int], float] = {}
    doc_lookup: Dict[Tuple[str, int], Document] = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            doc_lookup.setdefault(key, doc)

    fused_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return [doc_lookup[key] for key in fused_keys]


def hybrid_retrieve(
    collection_name: str,
    query: str,
    k: int = 6,
    fetch_k: int = 20,
    where_filter: Optional[Dict] = None,
) -> List[Document]:
    """
    Retrieve top-k documents by fusing BM25 keyword search with vector
    similarity search via RRF. Each retriever independently fetches up to
    fetch_k candidates before fusion narrows down to k.
    """
    vectorstore = get_vectorstore_for_query(collection_name)
    if vectorstore is None:
        return []

    try:
        if where_filter:
            vector_results = vectorstore.similarity_search(query, k=fetch_k, filter=where_filter)
        else:
            vector_results = vectorstore.similarity_search(query, k=fetch_k)
    except Exception:
        logger.exception("Vector search failed during hybrid retrieval for collection '%s'", collection_name)
        vector_results = []

    all_docs = get_all_documents(collection_name, where_filter=where_filter)
    bm25_results = bm25_search(all_docs, query, k=fetch_k)

    fused = reciprocal_rank_fusion([vector_results, bm25_results])
    return fused[:k]


def hybrid_retrieve_multi_query(
    collection_name: str,
    queries: List[str],
    k: int = 6,
    fetch_k: int = 20,
    where_filter: Optional[Dict] = None,
) -> List[Document]:
    """
    RAG-Fusion style multi-query hybrid retrieval: for each phrasing in
    `queries`, runs both a vector search and a BM25 search, then fuses
    every one of those rankings together via a single RRF pass. All
    phrasings are weighted equally — queries[0] isn't treated specially
    for scoring, only for logging.

    A single-element `queries` list behaves identically to hybrid_retrieve
    (same two rankings — one vector, one BM25 — fused the same way), so
    this is a strict generalization, not a different code path with
    different behavior at n=1.
    """
    if not queries:
        return []

    vectorstore = get_vectorstore_for_query(collection_name)
    if vectorstore is None:
        return []

    all_docs = get_all_documents(collection_name, where_filter=where_filter)
    bm25_index = build_bm25_index(all_docs)

    rankings: List[List[Document]] = []
    for q in queries:
        try:
            if where_filter:
                vector_results = vectorstore.similarity_search(q, k=fetch_k, filter=where_filter)
            else:
                vector_results = vectorstore.similarity_search(q, k=fetch_k)
        except Exception:
            logger.exception(
                "Vector search failed for query variant %r during multi-query retrieval "
                "for collection '%s'", q, collection_name,
            )
            vector_results = []
        rankings.append(vector_results)
        rankings.append(bm25_search_with_index(bm25_index, all_docs, q, fetch_k))

    fused = reciprocal_rank_fusion(rankings)
    return fused[:k]