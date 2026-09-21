
from core.ingestion import (
    hash_file_content,
    hash_chunk_content,
    build_chunk_header,
    get_splitter,
    ingest_files,
    get_ingestion_stats,
)


def test_hash_file_content_is_stable():
    content = "def foo():\n    return 1\n"
    assert hash_file_content(content) == hash_file_content(content)


def test_hash_file_content_changes_with_content():
    h1 = hash_file_content("def foo(): return 1")
    h2 = hash_file_content("def foo(): return 2")
    assert h1 != h2


def test_build_chunk_header_includes_path_and_language():
    header = build_chunk_header("app/main.py", "Python", role_hint="application entry point")
    assert "app/main.py" in header
    assert "Python" in header
    assert "application entry point" in header


def test_build_chunk_header_omits_role_when_none():
    header = build_chunk_header("README.md", "Markdown", role_hint=None)
    assert "# Role:" not in header


def test_get_splitter_uses_python_separators_for_python():
    splitter = get_splitter("Python")
    assert "\ndef " in splitter._separators


def test_get_splitter_falls_back_to_default_for_unknown_language():
    splitter = get_splitter("Unknown")
    assert splitter._separators == ["\n\n", "\n", " ", ""]


def test_ingest_files_produces_documents_with_expected_metadata():
    files = [{
        "path": "app/utils/math_helpers.py",
        "content": "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n",
        "size_kb": 0.1,
    }]

    docs = ingest_files(files)

    assert len(docs) >= 1
    for doc in docs:
        assert doc.metadata["source"] == "app/utils/math_helpers.py"
        assert doc.metadata["language"] == "Python"
        assert "content_hash" in doc.metadata
        assert doc.page_content.startswith("# File: app/utils/math_helpers.py")


def test_ingest_files_same_content_hash_across_chunks_of_one_file():
    # A single file's chunks should all share the same content_hash — the
    # hash is per-file (used for incremental re-index diffing), not per-chunk.
    long_content = "\n\n".join(f"def fn_{i}():\n    return {i}" for i in range(50))
    files = [{"path": "big.py", "content": long_content, "size_kb": 5}]

    docs = ingest_files(files)

    assert len(docs) > 1, "expected the long file to split into multiple chunks"
    hashes = {doc.metadata["content_hash"] for doc in docs}
    assert len(hashes) == 1


def test_ingest_files_skips_empty_content():
    files = [{"path": "empty.py", "content": "", "size_kb": 0}]
    docs = ingest_files(files)
    assert docs == []


def test_ingest_files_tags_chunk_method_ast_for_python():
    files = [{"path": "a.py", "content": "def a():\n    return 1\n", "size_kb": 0.1}]
    docs = ingest_files(files)
    assert all(doc.metadata["chunk_method"] == "ast" for doc in docs)


def test_ingest_files_tags_chunk_method_heuristic_for_unsupported_language():
    files = [{"path": "a.sql", "content": "SELECT * FROM users;\n", "size_kb": 0.1}]
    docs = ingest_files(files)
    assert all(doc.metadata["chunk_method"] == "heuristic" for doc in docs)


def test_get_ingestion_stats_empty():
    stats = get_ingestion_stats([])
    assert stats == {"total_chunks": 0, "unique_files": 0, "languages": {}, "chunk_methods": {}}


def test_get_ingestion_stats_counts_files_and_languages():
    files = [
        {"path": "a.py", "content": "def a(): pass", "size_kb": 0.1},
        {"path": "b.js", "content": "function b() {}", "size_kb": 0.1},
    ]
    docs = ingest_files(files)
    stats = get_ingestion_stats(docs)

    assert stats["unique_files"] == 2
    assert "Python" in stats["languages"]
    assert "JavaScript" in stats["languages"]

# ---------- hash_chunk_content ----------

def test_hash_chunk_content_is_stable():
    chunk = "def foo():\n    return 1\n"
    assert hash_chunk_content(chunk) == hash_chunk_content(chunk)


def test_hash_chunk_content_changes_with_content():
    h1 = hash_chunk_content("def foo(): return 1")
    h2 = hash_chunk_content("def foo(): return 2")
    assert h1 != h2


def test_hash_chunk_content_ignores_the_per_file_header():
    # hash_chunk_content is called on the raw chunk BEFORE build_chunk_header
    # prepends "# File: ..." — two files with identical code but different
    # paths should produce the same chunk hash.
    same_code = "def helper():\n    return 42\n"
    assert hash_chunk_content(same_code) == hash_chunk_content(same_code)


# ---------- chunk_content_hash metadata (dedup key) ----------

def test_ingest_files_tags_chunk_content_hash():
    files = [{"path": "a.py", "content": "def a():\n    return 1\n", "size_kb": 0.1}]
    docs = ingest_files(files)
    assert all("chunk_content_hash" in doc.metadata for doc in docs)


def test_ingest_files_identical_code_in_different_files_shares_chunk_content_hash():
    # Simulates a vendored/duplicated file: byte-identical content at two
    # different paths should get the same chunk_content_hash, even though
    # their page_content differs (different "# File: ..." header).
    shared_code = "def licensed_helper():\n    return 'boilerplate'\n"
    files = [
        {"path": "vendor/a/helper.py", "content": shared_code, "size_kb": 0.1},
        {"path": "vendor/b/helper.py", "content": shared_code, "size_kb": 0.1},
    ]

    docs = ingest_files(files)

    assert len(docs) == 2
    hashes = {doc.metadata["chunk_content_hash"] for doc in docs}
    assert len(hashes) == 1  # same chunk content -> same hash
    # but page_content still differs, since each carries its own file header
    assert docs[0].page_content != docs[1].page_content
    assert "vendor/a/helper.py" in docs[0].page_content
    assert "vendor/b/helper.py" in docs[1].page_content


def test_ingest_files_different_code_gets_different_chunk_content_hash():
    files = [
        {"path": "a.py", "content": "def a():\n    return 1\n", "size_kb": 0.1},
        {"path": "b.py", "content": "def b():\n    return 2\n", "size_kb": 0.1},
    ]
    docs = ingest_files(files)
    hashes = {doc.metadata["chunk_content_hash"] for doc in docs}
    assert len(hashes) == 2


# ---------- symbol_name / symbol_type / parent_symbol metadata ----------

def test_ingest_files_ast_chunked_function_gets_symbol_metadata():
    files = [{"path": "a.py", "content": "def foo(a, b):\n    return a + b\n", "size_kb": 0.1}]
    docs = ingest_files(files)

    assert len(docs) == 1
    assert docs[0].metadata["symbol_name"] == "foo"
    assert docs[0].metadata["symbol_type"] == "function"
    assert docs[0].metadata["parent_symbol"] == ""
    assert docs[0].metadata["chunk_method"] == "ast"


def test_ingest_files_ast_chunked_method_gets_parent_symbol():
    content = (
        "class Bar:\n"
        "    def method(self):\n"
        "        return 1\n"
    )
    files = [{"path": "a.py", "content": content, "size_kb": 0.1}]
    docs = ingest_files(files)

    method_doc = next(d for d in docs if "def method" in d.page_content)
    assert method_doc.metadata["symbol_name"] == "method"
    assert method_doc.metadata["symbol_type"] == "method"
    assert method_doc.metadata["parent_symbol"] == "Bar"


def test_ingest_files_heuristic_chunked_language_has_empty_symbol_fields():
    # SQL has no tree-sitter grammar wired up in ast_chunking.py -> always
    # falls back to the heuristic splitter, which can't identify symbols.
    files = [{"path": "a.sql", "content": "SELECT * FROM users;\n", "size_kb": 0.1}]
    docs = ingest_files(files)

    assert len(docs) >= 1
    for doc in docs:
        assert doc.metadata["symbol_name"] == ""
        assert doc.metadata["symbol_type"] == ""
        assert doc.metadata["parent_symbol"] == ""
        assert doc.metadata["chunk_method"] == "heuristic"


def test_ingest_files_symbol_metadata_never_none_for_chroma_compatibility():
    # chromadb silently drops None-valued metadata keys — every chunk must
    # carry a concrete (possibly empty-string) value for every field, so
    # the metadata schema stays uniform across the whole collection.
    files = [
        {"path": "a.py", "content": "def foo(): pass", "size_kb": 0.1},
        {"path": "b.py", "content": "import os\n", "size_kb": 0.1},  # no definition -> None internally
        {"path": "c.sql", "content": "SELECT 1;", "size_kb": 0.1},     # heuristic path
    ]
    docs = ingest_files(files)

    for doc in docs:
        assert doc.metadata["symbol_name"] is not None
        assert doc.metadata["symbol_type"] is not None
        assert doc.metadata["parent_symbol"] is not None