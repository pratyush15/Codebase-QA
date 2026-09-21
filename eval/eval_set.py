
"""
Hand-built retrieval eval set.

Each entry is a question a developer might actually ask about this
codebase, paired with the file(s) that contain the real answer — verified
by reading the source, not guessed. Ground truth is this project's own
code, so the eval is fully reproducible by anyone who clones the repo: no
external labeled dataset, no API dependency, just `python eval/run_eval.py`
against this same project.

expected_sources uses paths as they appear once indexed via
extract_files_from_directory(project_root) — i.e. relative to the project
root, matching the `source` field the retriever returns.
"""

EVAL_SET = [
    {
        "question": "How is zip-slip / path traversal prevented when extracting an uploaded zip file?",
        "expected_sources": {"app/utils/file_handler.py"},
        "expected_keywords": ['resolve', 'parents', 'UnsafeZipError'],
    },
    {
        "question": "How does the app decide whether to re-embed a file or skip it during re-indexing?",
        "expected_sources": {"app/core/indexing.py"},
        "expected_keywords": ['hash', 'unchanged', 'prior_hash'],
    },
    {
        "question": "Where is the content hash of a file computed?",
        "expected_sources": {"app/core/ingestion.py"},
        "expected_keywords": ['hash_file_content', 'sha256', 'hashlib'],
    },
    {
        "question": "How are code files split into chunks before embedding?",
        "expected_sources": {"app/core/ingestion.py", "app/core/ast_chunking.py"},
        "expected_keywords": ['AST', 'heuristic', 'tree-sitter'],
    },
    {
        "question": "How does the app split a class into a header chunk and separate method chunks?",
        "expected_sources": {"app/core/ast_chunking.py"},
        "expected_keywords": ['header', 'method', '_split_class_node'],
    },
    {
        "question": "What happens when tree-sitter parsing fails for a source file?",
        "expected_sources": {"app/core/ast_chunking.py"},
        "expected_keywords": ['fallback', 'heuristic', 'None'],
    },
    {
        "question": "How are BM25 keyword search and vector similarity search combined?",
        "expected_sources": {"app/core/hybrid_retriever.py"},
        "expected_keywords": ['reciprocal rank fusion', 'RRF', 'fuse'],
    },
    {
        "question": "How is reciprocal rank fusion implemented?",
        "expected_sources": {"app/core/hybrid_retriever.py"},
        "expected_keywords": ['reciprocal', 'rank', 'rrf_k'],
    },
    {
        "question": "What tokenizer is used for BM25 search over code?",
        "expected_sources": {"app/core/hybrid_retriever.py"},
        "expected_keywords": ['snake_case', 'identifier', 'tokenize'],
    },
    {
        "question": "How does cross-encoder reranking work in this project, and what happens if the reranker model can't be loaded?",
        "expected_sources": {"app/core/reranker.py"},
        "expected_keywords": ['cross-encoder', 'sentence-transformers', 'unavailable'],
    },
    {
        "question": "What retrieval modes are supported and how does the app choose between vector search, MMR, hybrid, and hybrid with reranking?",
        "expected_sources": {"app/core/retriever.py"},
        "expected_keywords": ['vector', 'mmr', 'hybrid', 'hybrid_rerank'],
    },
    {
        "question": "How is maximal marginal relevance (MMR) search implemented for reducing redundant chunks?",
        "expected_sources": {"app/core/retriever.py"},
        "expected_keywords": ['MMR', 'lambda_mult', 'diversity'],
    },
    {
        "question": "How does the app connect to and query the local Ollama LLM?",
        "expected_sources": {"app/core/qa_chain.py"},
        "expected_keywords": ['Ollama', 'OllamaLLM', 'base_url'],
    },
    {
        "question": "Where is query latency and token usage logged?",
        "expected_sources": {"app/core/qa_chain.py"},
        "expected_keywords": ['latency', 'logger', 'qa_chain'],
    },
    {
        "question": "How does the FastAPI layer expose the indexing and query functionality?",
        "expected_sources": {"app/api.py"},
        "expected_keywords": ['/index', '/query', 'FastAPI'],
    },
    {
        "question": "What happens in the API when Ollama is unreachable during a query?",
        "expected_sources": {"app/api.py"},
        "expected_keywords": ['503', 'unreachable', 'HTTPException'],
    },
    {
        "question": "How does the app connect to ChromaDB and create or load a collection?",
        "expected_sources": {"app/core/vectorstore.py"},
        "expected_keywords": ['PersistentClient', 'Chroma', 'collection'],
    },
    {
        "question": "How are all the chunk hashes for a collection retrieved, for incremental re-indexing?",
        "expected_sources": {"app/core/vectorstore.py"},
        "expected_keywords": ['content_hash', 'get_collection_file_hashes', 'metadata'],
    },
    {
        "question": "How does the sidebar component let a user upload a project and see indexing results?",
        "expected_sources": {"app/components/sidebar.py"},
        "expected_keywords": ['file_uploader', 'sync_files_to_collection', 'success'],
    },
    {
        "question": "How does the app detect the programming language of an uploaded file?",
        "expected_sources": {"app/utils/language_detect.py"},
        "expected_keywords": ['extension', 'language', 'detect_language'],
    },
    {
        "question": "What environment variables does the app require at startup, and what happens if one is missing?",
        "expected_sources": {"app/config.py"},
        "expected_keywords": ['OLLAMA_BASE_URL', '_require', 'ConfigError'],
    },
]