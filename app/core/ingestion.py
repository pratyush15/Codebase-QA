from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.language_detect import get_language_context, get_file_role_hint


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
      - metadata: path, language, fence, is_code, role, chunk_index
    """
    documents = []

    for file in files:
        path = file["path"]
        content = file["content"]

        lang_ctx = get_language_context(path)
        language = lang_ctx["language"]
        role_hint = get_file_role_hint(path)

        splitter = get_splitter(language)
        chunks = splitter.split_text(content)

        if not chunks:
            continue

        header = build_chunk_header(path, language, role_hint)

        for idx, chunk in enumerate(chunks):
            page_content = header + chunk

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
            }

            documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def get_ingestion_stats(documents: List[Document]) -> Dict:
    """
    Summary stats after ingestion — shown in the UI after upload.
    """
    if not documents:
        return {"total_chunks": 0, "unique_files": 0, "languages": {}}

    sources = set()
    languages = {}

    for doc in documents:
        sources.add(doc.metadata["source"])
        lang = doc.metadata["language"]
        languages[lang] = languages.get(lang, 0) + 1

    return {
        "total_chunks": len(documents),
        "unique_files": len(sources),
        "languages": dict(sorted(languages.items(), key=lambda x: -x[1])),
    }