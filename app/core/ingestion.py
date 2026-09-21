
import hashlib
from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.language_detect import get_language_context, get_file_role_hint
from core.ast_chunking import chunk_source_ast_with_symbols, supports_ast_chunking


# Language-aware separators for better code splitting
LANGUAGE_SEPARATORS = {
    "Python": ["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
    "JavaScript": ["\nfunction ", "\nclass ", "\nconst ", "\n\n", "\n", " ", ""],
    "TypeScript": ["\nfunction ", "\nclass ", "\nconst ", "\ninterface ", "\n\n", "\n", " ", ""],
    "JavaScript (React)": ["\nfunction ", "\nconst ", "\n\n", "\n", " ", ""],
    "TypeScript (React)": ["\nfunction ", "\nconst ", "\ninterface ", "\n\n", "\n", " ", ""],
    "Java": ["\npublic class ", "\nprivate ", "\npublic ", "\n\n", "\n", " ", ""],
    "Go": ["\nfunc ", "\ntype ", "\n\n", "\n", " ", ""],
    "Rust": ["\nfn ", "\nimpl ", "\nstruct ", "\n\n", "\n", " ", ""],
    "C++": ["\nvoid ", "\nint ", "\nclass ", "\n\n", "\n", " ", ""],
    "SQL": ["\nCREATE ", "\nSELECT ", "\nINSERT ", "\n\n", "\n", " ", ""],
}

DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]


def hash_file_content(content: str) -> str:
    """
    Stable content hash for a file, used to detect unchanged vs. changed
    files across re-indexing runs (see core.indexing.sync_files_to_collection).
    """
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def hash_chunk_content(chunk: str) -> str:
    """
    Stable content hash for a single chunk's raw text (pre-header — i.e.
    the code itself, not the "# File: ..." prefix build_chunk_header adds).
    Used by core.vectorstore.add_documents_to_collection to dedupe
    *embedding* calls: vendored copies of the same file, generated
    boilerplate, or a shared license header block all produce identical
    chunk text in more than one file. Hashing the raw chunk (not the
    per-file page_content) means two files with byte-identical code but
    different paths still get recognized as the same content, since the
    per-file header text is deliberately excluded from this hash.
    """
    return hashlib.sha256(chunk.encode("utf-8", errors="replace")).hexdigest()


def get_splitter(language: str) -> RecursiveCharacterTextSplitter:
    separators = LANGUAGE_SEPARATORS.get(language, DEFAULT_SEPARATORS)
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
    )


def build_chunk_header(file_path: str, language: str, role_hint: str = None) -> str:
    """
    Prepend a small context header to each chunk so the LLM
    always knows which file and language it's reading.
    """
    header = f"# File: {file_path}\n# Language: {language}\n"
    if role_hint:
        header += f"# Role: {role_hint}\n"
    header += "---\n"
    return header


def ingest_files(files: List[Dict]) -> List[Document]:
    """
    Takes a list of file dicts from file_handler and returns
    a flat list of LangChain Documents ready for embedding.

    Each Document has:
      - page_content: chunk header + code chunk
      - metadata: path, language, fence, is_code, role, chunk_index, and
        (for AST-chunked languages) symbol_name / symbol_type /
        parent_symbol — see build_chunk_header and the module docstring
        of core.ast_chunking for what populates each field.
    """
    documents = []

    for file in files:
        path = file["path"]
        content = file["content"]

        lang_ctx = get_language_context(path)
        language = lang_ctx["language"]
        role_hint = get_file_role_hint(path)
        content_hash = hash_file_content(content)

        # Normalize both chunking paths to the same shape: a list of dicts
        # with at least "text", so the loop below doesn't need to care
        # which splitter produced them. The heuristic path can't identify
        # symbols at all (it's just cutting on separator strings, not
        # walking a syntax tree), so its chunks carry symbol_name=None.
        chunks = None
        chunk_method = "heuristic"
        if supports_ast_chunking(language):
            chunks = chunk_source_ast_with_symbols(content, language, max_chunk_size=CHUNK_SIZE)
            if chunks is not None:
                chunk_method = "ast"

        if chunks is None:
            splitter = get_splitter(language)
            chunks = [
                {"text": text, "symbol_name": None, "symbol_type": None, "parent_symbol": None}
                for text in splitter.split_text(content)
            ]

        if not chunks:
            continue

        header = build_chunk_header(path, language, role_hint)

        for idx, chunk in enumerate(chunks):
            chunk_text = chunk["text"]
            page_content = header + chunk_text

            metadata = {
                "source": path,
                "language": language,
                "fence": lang_ctx["fence"],
                "is_code": lang_ctx["is_code"],
                "is_config": lang_ctx["is_config"],
                "is_markup": lang_ctx["is_markup"],
                "role": role_hint or "",
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "size_kb": file.get("size_kb", 0),
                "content_hash": content_hash,
                "chunk_content_hash": hash_chunk_content(chunk_text),
                "chunk_method": chunk_method,
                # "" rather than None: chromadb silently DROPS metadata
                # keys whose value is None on write (verified against the
                # pinned chromadb version) rather than storing a null —
                # which would make "symbol_name" present on some chunks
                # and absent on others, complicating any code (filters,
                # metadata scans) that assumes a uniform key set. An empty
                # string keeps the schema uniform and still reads as
                # "no symbol" everywhere it's checked.
                "symbol_name": chunk["symbol_name"] or "",
                "symbol_type": chunk["symbol_type"] or "",
                "parent_symbol": chunk["parent_symbol"] or "",
            }

            documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def get_ingestion_stats(documents: List[Document]) -> Dict:
    """
    Summary stats after ingestion — shown in the UI after upload.
    """
    if not documents:
        return {"total_chunks": 0, "unique_files": 0, "languages": {}, "chunk_methods": {}}

    sources = set()
    languages = {}
    chunk_methods = {}

    for doc in documents:
        sources.add(doc.metadata["source"])
        lang = doc.metadata["language"]
        languages[lang] = languages.get(lang, 0) + 1
        method = doc.metadata.get("chunk_method", "heuristic")
        chunk_methods[method] = chunk_methods.get(method, 0) + 1

    return {
        "total_chunks": len(documents),
        "unique_files": len(sources),
        "languages": dict(sorted(languages.items(), key=lambda x: -x[1])),
        "chunk_methods": chunk_methods,
    }