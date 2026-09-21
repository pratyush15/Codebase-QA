"""
Cross-encoder reranking.

The initial retrieval (vector similarity, or hybrid BM25+vector) uses a
bi-encoder: query and chunk are embedded independently, then compared by
cosine distance. That's fast enough to search a whole collection, but it's
a weaker relevance signal than letting a model attend over the query and
the chunk *together*.

A cross-encoder does exactly that — it scores each (query, chunk) pair
jointly. Too slow to run over an entire collection, but perfectly fine to
run over a short candidate list (fetch_k results from the first-pass
retriever) to re-order them by a more accurate relevance score before
handing the top few to the LLM.

Requires the optional `sentence-transformers` dependency. The model
(~90MB, cross-encoder/ms-marco-MiniLM-L-6-v2 by default) downloads from
Hugging Face on first use and is cached locally afterward — that first
call needs network access. If the package isn't installed, or the model
can't be loaded (e.g. no network yet), reranking is skipped and the
original ranking is returned unchanged rather than failing the query.
"""
import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed — reranking disabled. "
        "Install it (`pip install sentence-transformers`) to enable this stage."
    )

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model_cache = {}


def _get_model(model_name: str) -> "CrossEncoder":
    if model_name not in _model_cache:
        logger.info("Loading cross-encoder reranker model '%s' (first call may download it)...", model_name)
        _model_cache[model_name] = CrossEncoder(model_name)
    return _model_cache[model_name]


def is_reranking_available() -> bool:
    return SENTENCE_TRANSFORMERS_AVAILABLE


def rerank(
    query: str,
    docs: List[Document],
    top_k: int,
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> List[Document]:
    """
    Re-score `docs` against `query` with a cross-encoder and return the
    top_k, re-ordered by that score (highest relevance first).

    Falls back to `docs[:top_k]` unchanged, with a logged warning, if
    sentence-transformers isn't installed or the model fails to load —
    reranking is a quality improvement, not something a query should hard-
    fail over.
    """
    if not docs:
        return []
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return docs[:top_k]

    try:
        model = _get_model(model_name)
        pairs = [(query, doc.page_content) for doc in docs]
        scores = model.predict(pairs)
    except Exception:
        logger.exception("Reranking failed (model load or inference) — returning unranked top_k instead.")
        return docs[:top_k]

    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]