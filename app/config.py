import os
from dotenv import load_dotenv

load_dotenv()

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL")

# ChromaDB
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MAX_RETRIEVAL_DOCS = 6

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