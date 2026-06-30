# Codebase Q&A

A local RAG-based assistant for querying any codebase using natural language. Upload your project, ask questions, and get context-aware answers — all running locally with no API keys needed.

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Qwen 2.5 via Ollama (local) |
| Embeddings | nomic-embed-text via Ollama (local) |
| Vector DB | ChromaDB (persistent on disk) |
| RAG | LangChain |
| UI | Streamlit |

---

## Project Structure

```
codebase-qa/
├── app/
│   ├── main.py                  # Entry point
│   ├── config.py                # Central config (models, paths, chunk sizes)
│   ├── components/
│   │   ├── sidebar.py           # Upload, project selector, filters
│   │   ├── chat.py              # Streaming chat interface
│   │   └── file_tree.py        # Indexed file browser
│   ├── core/
│   │   ├── ingestion.py         # File chunking and metadata
│   │   ├── vectorstore.py       # ChromaDB operations
│   │   ├── retriever.py         # Similarity search and context assembly
│   │   └── qa_chain.py          # LangChain + Ollama QA pipeline
│   └── utils/
│       ├── file_handler.py      # Zip extraction, encoding detection
│       └── language_detect.py  # Language and role metadata per file
├── data/
│   └── chroma_db/               # Persistent vector storage (auto-created)
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 1. Clone and install

```bash
git clone <your-repo-url>
cd codebase-qa
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Pull the models

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### 3. Configure environment

The `.env` file is pre-configured with sensible defaults:

```env
OLLAMA_BASE_URL=
OLLAMA_MODEL=
OLLAMA_EMBEDDING_MODEL=
CHROMA_DB_PATH=
```

Swap `qwen2.5:3b` for a larger variant (e.g. `qwen2.5-coder:7b`) if your machine can handle it.

### 4. Run

```bash
streamlit run app/main.py
```

---

## Usage

1. **Name your project** — enter a project name in the sidebar (e.g. `my-fastapi-app`)
2. **Upload your code** — either a `.zip` of your folder or individual files
3. **Click Index Codebase** — files are chunked, embedded, and stored in ChromaDB
4. **Ask questions** — type in the chat and get streaming answers with source attribution

You can maintain multiple indexed projects and switch between them from the dropdown. Each project is isolated in its own ChromaDB collection.

---

## Features

- **Zip or file upload** — upload a zipped project folder or individual source files
- **Multi-project support** — index multiple codebases, switch between them instantly
- **Language-aware chunking** — splits on class/function boundaries for Python, JS, Java, Go, Rust, SQL, and more
- **Streaming responses** — token-by-token output with a live cursor
- **Source attribution** — every answer shows which files were used
- **File scope filter** — narrow queries to a specific file from the sidebar
- **Persistent storage** — indexed projects survive restarts; re-index only when code changes
- **File tree view** — browse all indexed files with language icons in the right panel
- **Auto-project switch** — after indexing, the new project is automatically selected

---

## Supported File Types

| Category | Extensions |
|---|---|
| Python | `.py` |
| JavaScript / TypeScript | `.js` `.ts` `.jsx` `.tsx` |
| Java / Kotlin / Scala | `.java` `.kt` `.scala` |
| C / C++ / C# | `.c` `.cpp` `.h` `.cs` |
| Go / Rust / Ruby / PHP | `.go` `.rs` `.rb` `.php` |
| Web | `.html` `.css` `.scss` |
| Data / Config | `.json` `.yaml` `.yml` `.toml` `.sql` |
| Docs / Scripts | `.md` `.txt` `.sh` `.bash` |

Automatically skips: `__pycache__`, `.git`, `node_modules`, `venv`, `dist`, `build`, binary files, and lock files.

---

## Configuration Reference

All settings are in `app/config.py` and read from `.env`:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5:3b` | LLM model for answering |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `CHROMA_DB_PATH` | `./data/chroma_db` | Vector DB storage path |
| `CHUNK_SIZE` | `1000` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `MAX_RETRIEVAL_DOCS` | `6` | Chunks retrieved per query |

---

## Important: Fresh Run Behaviour

ChromaDB persists all indexed data to disk under `data/chroma_db/`. This means if you re-upload a project with the same name without deleting the old collection first, **the new files get appended on top of the old chunks** — leading to duplicate or stale results.

**Always delete the project collection before re-indexing:**

1. Select the project from the "Active project" dropdown in the sidebar
2. Click the **🗑️ Delete** button next to it
3. Then re-upload and index your files fresh

Alternatively, use a new project name each time to avoid conflicts entirely.

---

## Known Issues

- **Ollama must be running** before starting the app. If the sidebar shows "Ollama unreachable", run `ollama serve` in a separate terminal.
- **First embedding is slow** — `nomic-embed-text` loads into memory on first use; subsequent uploads are faster.
- **Large codebases** — indexing projects over ~500 files may take a few minutes depending on your machine.