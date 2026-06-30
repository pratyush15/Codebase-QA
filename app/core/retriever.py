from typing import List, Optional, Dict
from langchain_core.documents import Document

from config import MAX_RETRIEVAL_DOCS
from core.vectorstore import get_vectorstore_for_query


def retrieve_documents(
    collection_name: str,
    query: str,
    k: int = MAX_RETRIEVAL_DOCS,
    filter_language: Optional[str] = None,
    filter_source: Optional[str] = None,
) -> List[Document]:
    vectorstore = get_vectorstore_for_query(collection_name)
    if vectorstore is None:
        return []

    where_filter = _build_filter(filter_language, filter_source)

    try:
        if where_filter:
            return vectorstore.similarity_search(query, k=k, filter=where_filter)
        return vectorstore.similarity_search(query, k=k)
    except Exception:
        return []


def _build_filter(
    filter_language: Optional[str],
    filter_source: Optional[str],
) -> Optional[Dict]:
    conditions = []
    if filter_language:
        conditions.append({"language": {"$eq": filter_language}})
    if filter_source:
        conditions.append({"source": {"$eq": filter_source}})

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
        })

    return sources