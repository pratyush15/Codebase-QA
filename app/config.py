
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# The directory containing app/ — anchor point for resolving relative paths
# below, so config values don't silently change meaning depending on which
# directory a command happens to be run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Configure logging once, centrally, before anything else in the app runs.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid at startup."""


def _require(var_name: str) -> str:
    """
    Read a required env var, or fail loudly and immediately.

    Without this, a missing .env value (e.g. OLLAMA_BASE_URL) silently
    becomes None and only surfaces as a confusing error deep inside a
    request — instead we want the app to refuse to start.
    """
    value = os.getenv(var_name)
    if not value or not value.strip():
        raise ConfigError(
            f"Missing required environment variable '{var_name}'. "
            f"Check your .env file against .env.example."
        )
    return value.strip()


try:
    # Ollama
    OLLAMA_BASE_URL = _require("OLLAMA_BASE_URL")
    OLLAMA_MODEL = _require("OLLAMA_MODEL")
    OLLAMA_EMBEDDING_MODEL = _require("OLLAMA_EMBEDDING_MODEL")

    # ChromaDB
    CHROMA_DB_PATH = _require("CHROMA_DB_PATH")
except ConfigError as e:
    logger.error(str(e))
    sys.exit(f"❌ Startup aborted — {e}")

# A relative CHROMA_DB_PATH (e.g. "./data/chroma_db", the .env.example
# default) used to resolve against the process's current working
# directory — meaning `streamlit run app/main.py` from the project root,
# `python eval/run_eval.py` from the project root, and a one-off
# `python -c "..."` run from inside app/ would each silently read/write a
# *different* Chroma folder, depending only on cwd. Anchoring relative
# paths at PROJECT_ROOT instead makes the resolved path the same no
# matter where a command is launched from. An absolute path in .env is
# left untouched.
if not os.path.isabs(CHROMA_DB_PATH):
    CHROMA_DB_PATH = str((PROJECT_ROOT / CHROMA_DB_PATH).resolve())

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MAX_RETRIEVAL_DOCS = 6

# Embedding (indexing pipeline)
# Documents are embedded in batches of EMBEDDING_BATCH_SIZE per Ollama
# call; up to MAX_EMBEDDING_WORKERS batches are embedded concurrently
# (each is an independent HTTP round trip with no shared state), while
# writes to Chroma stay single-threaded regardless — see
# core.vectorstore.add_documents_to_collection. Set MAX_EMBEDDING_WORKERS=1
# to fall back to strictly sequential embedding (e.g. if your Ollama
# instance is CPU-bound and single-threaded, concurrency mostly just adds
# scheduling overhead with little to no wall-clock benefit).
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
MAX_EMBEDDING_WORKERS = int(os.getenv("MAX_EMBEDDING_WORKERS", "4"))

# Retrieval (Tier 3)
# "vector"        — plain top-k cosine similarity (original behavior)
# "mmr"           — vector search with maximal marginal relevance (less redundant chunks)
# "hybrid"        — BM25 + vector fused via reciprocal rank fusion
# "hybrid_rerank" — hybrid, then re-ordered by a cross-encoder before truncating to k
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid_rerank")
RETRIEVAL_FETCH_K = 20       # candidates each first-pass retriever pulls before fusion/MMR/rerank narrows to MAX_RETRIEVAL_DOCS
MMR_LAMBDA = 0.5             # 0 = max diversity, 1 = pure relevance (no diversity pressure)
RRF_K = 60                   # reciprocal rank fusion damping constant (standard default)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Conversation-aware query rewriting: resolves follow-up questions
# ("what about the class version?") into standalone queries before
# retrieval, using recent chat history. No-op (and no extra LLM call)
# when there's no history yet, i.e. on the first question in a chat.
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "true").strip().lower() == "true"
QUERY_REWRITE_HISTORY_TURNS = 3   # trailing (user, assistant) turn-pairs considered for rewriting/context

# Multi-query expansion (RAG-Fusion) for hybrid/hybrid_rerank retrieval:
# generates a few LLM paraphrases of the (already-standalone) query and
# fuses retrieval results across all of them, since a user's wording
# rarely matches the actual identifiers in the code that answers it.
# Defaults OFF — unlike query rewriting (which only fires on follow-ups
# and is often skipped entirely), this adds one full extra LLM call on
# *every* query, which is a real latency cost worth opting into
# deliberately rather than paying by default.
ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "false").strip().lower() == "true"
QUERY_EXPANSION_COUNT = int(os.getenv("QUERY_EXPANSION_COUNT", "2"))   # paraphrases generated per query

# Supported code extensions
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".html", ".css", ".scss", ".json", ".yaml", ".yml", ".toml",
    ".md", ".txt", ".sh", ".bash", ".sql", ".r", ".m", ".lua"
}

# Extensions to always skip
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf",
    ".zip", ".tar", ".gz", ".exe", ".bin", ".lock", ".pyc"
}

SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    "dist", "build", ".idea", ".vscode"
}

# Whether to additionally exclude files matched by the project's own root
# .gitignore during indexing (on top of the hardcoded SKIP_DIRS/SKIP_EXTENSIONS
# above). Keeps generated code, vendored deps, and project-specific build
# output out of the index without needing to hardcode every possible case.
RESPECT_GITIGNORE = os.getenv("RESPECT_GITIGNORE", "true").strip().lower() == "true"

# Upload / zip-extraction safety limits.
#
# MAX_UPLOAD_SIZE_MB bounds the raw upload itself, checked while reading
# the request body in chunks (see api._read_upload_capped) — a client
# can't force an unbounded amount of memory to be allocated just by
# sending a huge file.
#
# The rest bound what a *small* zip can do once extracted — the classic
# "zip bomb" problem, where a tiny archive expands to gigabytes of disk
# and hours of chunking/embedding work:
#   MAX_ZIP_UNCOMPRESSED_MB — total size of every entry's uncompressed
#     content, checked from the zip's central directory BEFORE extracting
#     a single byte.
#   MAX_ZIP_ENTRIES — total number of entries, since a "quantity bomb"
#     (e.g. a million empty files) is a resource-exhaustion problem even
#     at a small total byte size (metadata processing, os.walk, per-file
#     hashing/chunking all scale with file count, not just bytes).
#   MAX_ZIP_COMPRESSION_RATIO — any single entry compressing this many
#     times smaller than its uncompressed size is a strong zip-bomb
#     signal (e.g. the infamous "42.zip" achieves ~4.5 billion:1) and is
#     rejected outright, even if the archive would otherwise fit under
#     the total-size cap.
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))
MAX_ZIP_UNCOMPRESSED_MB = int(os.getenv("MAX_ZIP_UNCOMPRESSED_MB", "500"))
MAX_ZIP_ENTRIES = int(os.getenv("MAX_ZIP_ENTRIES", "20000"))
MAX_ZIP_COMPRESSION_RATIO = int(os.getenv("MAX_ZIP_COMPRESSION_RATIO", "100"))

# Query telemetry (see core/query_log.py): latency/token/hit-rate metrics
# for every completed query, persisted as JSONL so /metrics (api.py) can
# aggregate them ("avg latency by retrieval mode", "hit-rate over time")
# instead of grepping the plain-text logger.info() line qa_chain.py also
# emits. Records numeric/categorical telemetry only — never the question
# text, generated answer, or retrieved content (see query_log.py's module
# docstring). Lives under data/ (already gitignored) since it's a runtime
# artifact, not something to commit — same treatment as the Chroma DB itself.
ENABLE_QUERY_METRICS_LOG = os.getenv("ENABLE_QUERY_METRICS_LOG", "true").strip().lower() == "true"
QUERY_METRICS_LOG_PATH = os.getenv("QUERY_METRICS_LOG_PATH", "")
if not QUERY_METRICS_LOG_PATH:
    QUERY_METRICS_LOG_PATH = str(PROJECT_ROOT / "data" / "query_metrics.jsonl")
elif not os.path.isabs(QUERY_METRICS_LOG_PATH):
    QUERY_METRICS_LOG_PATH = str((PROJECT_ROOT / QUERY_METRICS_LOG_PATH).resolve())