import os
import re
from typing import List, Optional, Dict

import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from config import CHROMA_DB_PATH, OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL


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


def list_collections() -> List[Dict]:
    client = get_chroma_client()
    result = []
    for col in client.list_collections():
        try:
            count = client.get_collection(col.name).count()
        except Exception:
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


def add_documents_to_collection(collection_name: str, documents: list) -> int:
    vectorstore = create_or_load_vectorstore(collection_name)
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        vectorstore.add_documents(documents[i: i + batch_size])
    return get_chroma_client().get_collection(collection_name).count()


def delete_collection(collection_name: str) -> bool:
    try:
        get_chroma_client().delete_collection(collection_name)
        return True
    except Exception:
        return False


def get_collection_sources(collection_name: str) -> List[str]:
    try:
        col = get_chroma_client().get_collection(collection_name)
        result = col.get(include=["metadatas"])
        sources = {m["source"] for m in result.get("metadatas", []) if m and "source" in m}
        return sorted(sources)
    except Exception:
        return []


def get_vectorstore_for_query(collection_name: str) -> Optional[Chroma]:
    if not collection_exists(collection_name):
        return None
    return create_or_load_vectorstore(collection_name)