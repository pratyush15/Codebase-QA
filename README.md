# Codebase Q&A

Ask questions about a codebase in plain English and get answers grounded in the actual source — with citations back to the exact files and functions used. Everything runs locally: local LLM (Ollama), local embeddings, local vector store (ChromaDB). No code ever leaves your machine.

```
"How does the app decide whether to re-embed a file or skip it during re-indexing?"

  -> retrieves core/indexing.py (relevant chunks)
  -> answers using only that code, with the source file cited
```

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Architecture](#architecture)
- [Features](#features)
  - [Language-aware, AST-based chunking](#1-language-aware-ast-based-chunking)
  - [Incremental re-indexing (content hashing)](#2-incremental-re-indexing-content-hashing)
  - [Hybrid retrieval: BM25 + vector + RRF](#3-hybrid-retrieval-bm25--vector--reciprocal-rank-fusion)
  - [Cross-encoder reranking](#4-cross-encoder-reranking)
  - [Maximal Marginal Relevance (MMR)](#5-maximal-marginal-relevance-mmr)
  - [Conversation-aware query rewriting & multi-query expansion](#6-conversation-aware-query-rewriting--multi-query-expansion-rag-fusion)
  - [Retrieval evaluation harness](#7-retrieval-evaluation-harness)
  - [Answer-quality evaluation: keyword scoring & LLM-as-judge](#8-answer-quality-evaluation-keyword-scoring--llm-as-judge)
  - [FastAPI layer](#9-fastapi-layer)
  - [Streamlit UI](#10-streamlit-ui)
  - [Upload safety: zip-slip & zip-bomb protection](#11-upload-safety-zip-slip--zip-bomb-protection)
  - [.gitignore-aware indexing](#12-gitignore-aware-indexing)
  - [Concurrent embedding](#13-concurrent-embedding)
  - [Observability: query telemetry & /metrics](#14-observability-query-telemetry--metrics)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the app](#running-the-app)
  - [Streamlit UI](#streamlit-ui)
  - [FastAPI backend](#fastapi-backend)
  - [Docker](#docker)
- [Running the tests](#running-the-tests)
- [Running the evaluation harness](#running-the-evaluation-harness)
- [Project structure](#project-structure)
- [Supported languages](#supported-languages)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

---

## Why this exists

Dropping an entire codebase into an LLM's context window doesn't scale, and generic chunking (splitting code every N characters) routinely cuts a function in half — which then gets embedded and retrieved as two disconnected, useless fragments. This project builds a **retrieval-augmented generation (RAG) pipeline purpose-built for source code**: it parses code with real language grammars (tree-sitter) so chunks are always whole functions or methods, resolves follow-up questions and expands vocabulary before retrieval, retrieves with a hybrid of keyword and semantic search, reranks with a cross-encoder, and answers using a local LLM — with every claim traceable back to a specific file, and every stage of the pipeline measured rather than assumed.

It ships with two complementary measurement tools: a **retrieval evaluation harness** with a hand-built, self-referential question set (questions about this project, answered by this project) so retrieval quality is a number, not a vibe — plus optional **answer-quality scoring** (keyword coverage and LLM-as-judge) so "did we find the right file" and "did the model actually answer correctly" are tracked as two separate, comparable signals, with every run logged so a config change's effect is a diff against history, not something you have to remember.

---

## Architecture

```
                    +--------------+      +----------+
                    | Streamlit UI |      | FastAPI  |
                    | (main.py)    |      | (api.py) |
                    +--------------+      +----------+
                            |                   |
                            +---------+---------+
                                    |
                                    v
              +-------------------------------------------+
              |  INDEXING PIPELINE                        |
              |                                           |
              |  1. Upload (zip / files)                  |
              |  2. Zip-slip + zip-bomb safe extraction   |
              |     (utils/file_handler.py)               |
              |  3. .gitignore-aware filtering             |
              |     (utils/gitignore_filter.py)           |
              |  4. Incremental sync (core/indexing.py)   |
              |     content-hash diff ->                  |
              |     new / changed / unchanged / deleted   |
              |     (only new + changed files continue)   |
              |  5. Chunking (core/ast_chunking.py)       |
              |     tree-sitter -> whole functions        |
              |     & methods, per language, tagged with  |
              |     symbol_name/symbol_type/parent_symbol |
              |     falls back to heuristic splitter      |
              |     for unsupported languages             |
              |  6. Embedding (concurrent batches)        |
              |     Ollama embeddings (nomic-embed-text)  |
              +-------------------------------------------+
                                    |
                                    v
                           +------------------+
                           |  ChromaDB        |
                           |  (vector store)  |
                           +------------------+
                                    |
                                    v
             +----------------------------------------------+
             |  QUERY PRE-PROCESSING (core/query_rewrite.py) |
             |                                                |
             |  1. Standalone rewrite (if chat history)      |
             |  2. Multi-query expansion (optional, RAG-Fusion) |
             +----------------------------------------------+
                                    |
                                    v
          +---------------------------------------------------+
          |  RETRIEVAL PIPELINE (core/retriever.py)           |
          |                                                   |
          |  mode = retrieval_mode                            |
          |  optional filters: language / source file / symbol|
          |                                                   |
          |  "vector"        -> vector similarity search      |
          |  "mmr"           -> MMR (diversity-aware) search  |
          |  "hybrid"        -> BM25 + vector -> RRF fusion   |
          |  "hybrid_rerank" -> BM25 + vector -> RRF fusion   |
          |                     -> cross-encoder reranking    |
          +---------------------------------------------------+
                                    |
                                    v
                +----------------------------------------+
                |  ANSWER GENERATION (core/qa_chain.py)  |
                |                                        |
                |  Ollama LLM (qwen2.5:3b)               |
                |  -> latency / token metrics logged     |
                |  -> query_log.py: JSONL telemetry      |
                +----------------------------------------+
                                    |
                        answer + sources + metrics
                                    |
                +-------------------+-------------------+
                v                   v                   v
     +-----------------+  +-----------------+  +-------------------+
     | Streamlit UI    |  | FastAPI /query  |  | FastAPI /query/    |
     | (chat panel)    |  | (buffered JSON) |  | stream (SSE)       |
     +-----------------+  +-----------------+  +-------------------+


             +---------------------------------------------+
             |  EVALUATION (eval/)                         |
             |                                             |
             |  21-question eval set (eval_set.py)         |
             |    -> runs through the same retrieval       |
             |       pipeline shown above                  |
             |    -> precision@k, recall@k, MRR, hit-rate  |
             |       (metrics.py)                          |
             |    -> optional: full generation + keyword   |
             |       coverage scoring (answer_metrics.py)  |
             |    -> optional: LLM-as-judge grading         |
             |       (llm_judge.py)                        |
             |    -> every run persisted + diffed against  |
             |       history (eval_log.py)                 |
             +---------------------------------------------+
```

**Data flow, in words:**

1. **Upload** — a zip (or individual files) comes in through the Streamlit UI or `POST /index`.
2. **Safe extraction** — zip contents are extracted with a path-traversal check (zip-slip protection) and size/entry-count/compression-ratio limits (zip-bomb protection) before anything touches disk.
3. **.gitignore filtering** — if the project has a root `.gitignore`, matching files (build output, vendored deps, generated code) are excluded from indexing, on top of the hardcoded skip-list.
4. **Incremental sync** — each file's SHA-256 content hash is compared against what's already indexed; only new or changed files proceed to chunking, and (on a full sync) files removed from the project have their old chunks deleted too.
5. **Chunking** — supported languages (Python, JS/TS, Java, Go, Rust, C/C++) are parsed with tree-sitter and split along real function/class boundaries, each chunk tagged with the symbol it defines; everything else falls back to a heuristic separator-based splitter.
6. **Embedding** — chunks are embedded in concurrent batches with a local Ollama embedding model and stored in ChromaDB, tagged with source file, language, chunk index, content hash, and symbol metadata.
7. **Query pre-processing** — a follow-up question is rewritten into a standalone one using recent chat history (if needed), and optionally expanded into a few alternate phrasings (RAG-Fusion) before retrieval.
8. **Retrieval** — the (possibly rewritten/expanded) question is retrieved using one of four configurable modes, optionally scoped by language, source file, or symbol name.
9. **Answer** — the retrieved chunks are assembled into a prompt and streamed back from a local Ollama LLM, with sources cited, latency/token metrics logged to both the app log and a queryable JSONL telemetry file.

---

## Features

### 1. Language-aware, AST-based chunking

**Problem it solves:** naive text splitting (cut every N characters, or on blank lines) frequently slices a function in half. Half a function embedded and retrieved on its own is close to useless — the model sees a fragment with no return statement, or a method body with no signature.

**How it works** (`core/ast_chunking.py`): each supported file is parsed into a real syntax tree with [tree-sitter](https://tree-sitter.github.io/tree-sitter/). Top-level functions and classes become individual chunks by walking the tree and slicing along actual node boundaries — never mid-statement. Classes are chunked further: a "header" chunk (signature, docstring, fields) plus one chunk per method, each tagged with a `# (method of ClassName)` comment so context isn't lost once methods are split apart from their class body.

Every AST-derived chunk also carries `symbol_name` (the function/class/method it defines), `symbol_type` (`function` / `class` / `method`), and `parent_symbol` (the enclosing class, for methods) in its metadata. This is what powers the **"scope to symbol"** filter in retrieval — asking about the `retrieve_documents` function specifically, rather than the whole file. Only AST-derived chunks carry these fields; heuristically-split chunks never match a symbol filter.

Any chunk that's still unreasonably large after AST splitting (e.g. one enormous generated function) is passed through the heuristic splitter as a safety net, so nothing ever blows past a sane size.

**Fallback:** unsupported languages (SQL, HTML, YAML, Markdown, shell, etc.) — or any file that fails to parse — silently fall back to the original heuristic `RecursiveCharacterTextSplitter`. Nothing breaks; you just don't get the whole-function guarantee (or symbol metadata) for those files. Every chunk is tagged with `chunk_method: "ast"` or `"heuristic"` in its metadata, and the sidebar shows the breakdown after indexing.

**Supported for AST chunking:** Python, JavaScript, JSX, TypeScript, TSX, Java, Go, Rust, C, C++.

---

### 2. Incremental re-indexing (content hashing)

**Problem it solves:** re-uploading the same (or a slightly changed) project used to always re-embed and re-add every file, silently duplicating chunks for anything already indexed.

**How it works** (`core/indexing.py`): every file gets a SHA-256 hash of its content, stored alongside its chunks. On re-index, `sync_files_to_collection()` compares the new upload's hashes against what's already in the collection:

- **Unchanged** — hash matches → skipped entirely, no re-embedding.
- **Changed** — hash differs → old chunks for that file are deleted, new ones are embedded and added.
- **New** — no prior hash → embedded and added.
- **Deleted** — present in the collection but absent from a **full sync** upload → its chunks are removed too.

`sync_files_to_collection()` takes a `full_sync` flag (surfaced in the Streamlit sidebar as an upload-mode choice, and in the API as `POST /index`'s `full_sync` form field, default `true`). A full sync treats the upload as the complete, current state of the project, so anything missing is deleted. A partial sync (`full_sync=false`) is a deliberate subset upload — e.g. just a `src/` subfolder — where anything not included is left untouched rather than deleted.

---

### 3. Hybrid retrieval: BM25 + vector + Reciprocal Rank Fusion

**Problem it solves:** pure vector (embedding) similarity search is a *semantic* signal — it's good at "this chunk is conceptually related" but frequently under-ranks a chunk that contains the *exact* identifier, variable name, or error string being searched for, because the surrounding words differ. Pure keyword search has the opposite problem: no understanding of synonyms or paraphrasing.

**How it works** (`core/hybrid_retriever.py`):
- **BM25** — a classic keyword ranking algorithm ([`rank_bm25`](https://github.com/dorianbrown/rank_bm25)), run over a code-aware tokenizer that also splits `snake_case` identifiers into their component words (so a query like *"find user by id"* matches a chunk containing `find_user_by_id`).
- **Vector search** — standard cosine-similarity search over the Ollama embeddings already stored in Chroma.
- **Reciprocal Rank Fusion (RRF)** — the two ranked lists are merged without needing their raw scores to be on comparable scales: each chunk's fused score is `sum(1 / (rrf_k + rank))` across every ranking it appears in. A chunk both retrievers agree on floats to the top.
- **Multi-query fusion** (`hybrid_retrieve_multi_query()`) — the same BM25 + vector + RRF machinery, run across the original query *and* every expanded paraphrase (see [query rewriting & expansion](#6-conversation-aware-query-rewriting--multi-query-expansion-rag-fusion)), with all rankings fused together in one RRF pass.

---

### 4. Cross-encoder reranking

**Problem it solves:** both vector search and BM25 are *bi-encoder*-style retrieval — query and chunk are scored independently, which is fast enough to search a whole collection but is a weaker relevance signal than letting a model actually look at the query and the candidate chunk *together*.

**How it works** (`core/reranker.py`): after hybrid retrieval produces a shortlist of ~20 candidates, a [cross-encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html) (`cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`) scores each `(query, chunk)` pair jointly and re-orders the shortlist by that score before truncating to the final top-k sent to the LLM.

**Optional & fail-soft:** `sentence-transformers` is an optional dependency. If it isn't installed, or the model can't be loaded (e.g. no network on first run — it downloads from Hugging Face and caches locally afterward), reranking is skipped with a logged warning and the unranked hybrid result is used instead. A query never hard-fails because of the reranker.

---

### 5. Maximal Marginal Relevance (MMR)

**Problem it solves:** plain top-k similarity search can return 6 near-duplicate chunks from the same function or file, wasting context budget on redundant information instead of covering the question from multiple angles.

**How it works:** uses ChromaDB's native `max_marginal_relevance_search`, which re-ranks candidates to balance relevance against diversity — penalizing a candidate for being too similar to one already selected. Tunable via `MMR_LAMBDA` (`0` = maximum diversity, `1` = pure relevance, no diversity pressure).

---

### 6. Conversation-aware query rewriting & multi-query expansion (RAG-Fusion)

**Problem it solves:** two separate gaps between "what the user typed" and "what retrieval needs":
1. A follow-up question like *"what about the class version?"* shares almost no vocabulary with the code it's actually asking about — retrieved on its own, it matches nothing useful.
2. Even a standalone, well-formed question often uses different words than the code that answers it (*"log a user in"* vs. a codebase that says `authenticate_session`).

**How it works** (`core/query_rewrite.py`):
- **Standalone rewriting** (`rewrite_query_standalone()`) — given the last few turns of chat history (`QUERY_REWRITE_HISTORY_TURNS`, default 3 turn-pairs), an LLM call resolves pronouns and references ("that", "it", "the class version") into a fully standalone question before retrieval. A no-op (and no extra LLM call at all) on the first question in a chat, since there's no history to resolve against.
- **Multi-query expansion** (`expand_query()`) — asks the LLM for a few alternative phrasings (`QUERY_EXPANSION_COUNT`, default 2) of the already-standalone question. `core.hybrid_retriever.hybrid_retrieve_multi_query()` then retrieves against the original *and* every paraphrase, fusing all the rankings together — a chunk that only matches one phrasing's wording still gets a fair shot.

**Toggles:** query rewriting is `ENABLE_QUERY_REWRITE` (default **on**) — cheap, since it only fires on real follow-ups. Query expansion is `ENABLE_QUERY_EXPANSION` (default **off**) — it costs one full extra LLM call on *every* query, so it's opt-in rather than a default latency tax.

Both the rewritten query and the expanded paraphrases (plus their own latencies) are surfaced back in the response's `metrics` object (`query_rewritten`, `rewritten_query`, `expanded_queries`, and their `*_latency_ms` fields) — so a caller can see exactly what was actually searched for, not just what was typed.

---

### 7. Retrieval evaluation harness

**Problem it solves:** without measurement, "I improved retrieval" is a guess. This gives it a number.

**How it works** (`eval/`):
- `eval/eval_set.py` — 21 hand-written questions about *this project's own code*, each paired with the exact file(s) that contain the real answer, verified by reading the source. Because the ground truth is this repo, the eval is fully reproducible by anyone who clones it — no external labeled dataset needed.
- `eval/metrics.py` — pure, fully unit-tested scoring functions: **precision@k** (of the top-k retrieved chunks, how many are actually relevant), **recall@k** / hit-rate (was a relevant file found at all), and **MRR** (mean reciprocal rank — rewards ranking the relevant result *near the top*, not just including it somewhere).
- `eval/run_eval.py` — CLI that indexes the target project, runs every question against one or more retrieval modes, and prints a comparison table.

```bash
python eval/run_eval.py --modes vector mmr hybrid hybrid_rerank
```

```
Retrieval eval — 21 questions, k=6

mode            precision@k   recall@k    MRR       hit rate
----------------------------------------------------------------
vector          0.31          0.71        0.58      0.71
mmr             0.29          0.76        0.61      0.76
hybrid          0.34          0.86        0.74      0.86
hybrid_rerank   0.38          0.90        0.81      0.90
```
*(illustrative — run it yourself to get real numbers for your machine/models)*

Every run is logged with a timestamp (see [feature 8](#8-answer-quality-evaluation-keyword-scoring--llm-as-judge) below) and diffed against the most recent prior run for the same mode(s), so a config change's effect on these numbers is visible immediately rather than requiring you to remember the old baseline.

---

### 8. Answer-quality evaluation: keyword scoring & LLM-as-judge

**Problem it solves:** the retrieval eval above only answers "did we find the right file?" — it says nothing about whether the LLM actually used that file correctly to produce a correct answer. A retrieval score can look great while the model still hallucinates.

**How it works** (`eval/answer_metrics.py`, `eval/llm_judge.py`, `eval/run_eval.py --score-answers` / `--judge`):
- **`--score-answers`** runs the *full* retrieval → generation pipeline (the same code path a real user hits) for each eval question and scores the generated answer against `eval_set.py`'s `expected_keywords` — `keyword_coverage()` (what fraction of expected keywords appear) and `keyword_hit()` (whether coverage clears a minimum threshold, default 50%). Fast, deterministic, no extra LLM call.
- **`--judge`** additionally has a *second* LLM call grade each generated answer `correct` / `partial` / `incorrect` against the question (`eval/llm_judge.py`) — catching a wrong or hallucinated explanation that happens to still namedrop the right identifiers, which keyword matching alone can't tell apart from a genuinely correct answer. `--judge` implies `--score-answers` (there's no answer to judge without generating one first). This roughly doubles per-question eval cost and adds the judge model's own run-to-run noise, being an LLM call itself.
- Answer scoring runs at the same `--k` as the retrieval-only pass (defaulting to `config.MAX_RETRIEVAL_DOCS`), so one run's report describes a single consistent retrieval depth throughout.

**Run history** (`eval/eval_log.py`): every run is persisted as JSON Lines (timestamp + a snapshot of the retrieval/chunking config in effect — model names, chunk size, retrieval tuning) so "did that chunk-size change / reranker swap help or hurt?" is answerable by comparing logged runs, not by re-running an old config from memory.

```bash
python eval/run_eval.py --score-answers --judge
python eval/run_eval.py --history            # print the last 10 logged runs and exit
python eval/run_eval.py --history 30         # print the last 30
python eval/run_eval.py --no-log             # run without appending to the log (throwaway check)
```

---

### 9. FastAPI layer

**Problem it solves:** the original app was entirely coupled to Streamlit's `session_state`, meaning indexing and querying could only ever happen from inside a running Streamlit session.

**How it works** (`api.py`): a stateless REST API over the same `core/` logic the UI uses — nothing in it touches `st.session_state`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Ollama connectivity check |
| `GET` | `/collections` | List indexed projects with chunk counts |
| `GET` | `/collections/{name}/sources` | List indexed source files in a collection |
| `DELETE` | `/collections/{name}` | Delete a collection |
| `GET` | `/metrics` | Aggregated query telemetry (latency percentiles, hit-rate, token estimates), overall and by retrieval mode — optional `since_minutes` / `collection` query params |
| `POST` | `/index` | Upload a zip (`multipart/form-data`: `project_name`, `file`, optional `full_sync`) — indexes incrementally |
| `POST` | `/query` | Ask a question (JSON body) — returns the full buffered answer, cited sources, and metrics |
| `POST` | `/query/stream` | Same request body as `/query`, but responds with a Server-Sent Events stream (`token` / `sources` / `metrics` / `error` / `done` events) instead of one buffered response |

`/query` and `/query/stream` both accept optional `filter_language`, `filter_source`, `filter_symbol` (scope retrieval to a language, a specific file, or a specific function/class/method by name), `retrieval_mode` (override the server default), and `chat_history` (prior turns, oldest first, for conversation-aware query rewriting).

`/index` and `/query` return a clean `503` with a helpful message (rather than a raw stack trace) if Ollama is unreachable. Interactive docs are auto-generated at `/docs` once the server is running.

---

### 10. Streamlit UI

The original interface (`main.py`, `components/`): upload a project (zip or individual files), pick a collection from the sidebar, ask questions in a chat panel, see cited sources and a file tree of what's indexed. Shows incremental sync stats (new/updated/unchanged/deleted file counts) and the AST-vs-heuristic chunking breakdown after every index run. The sidebar also lets you scope a question to a specific file or symbol before asking, and delete a collection outright.

---

### 11. Upload safety: zip-slip & zip-bomb protection

**Problem it solves:** naively calling `zipfile.extractall()` on an untrusted archive is vulnerable to [zip-slip](https://snyk.io/research/zip-slip-vulnerability) — a malicious entry named e.g. `../../../etc/evil.py` can write outside the intended extraction directory. Separately, a tiny, entirely well-behaved-looking zip can still be a **zip bomb** — a small archive that expands to gigabytes on disk, or contains millions of entries — turning a routine upload into a resource-exhaustion attack.

**How it works** (`utils/file_handler.py`, `api.py`):
- **Zip-slip** — every archive member's resolved path is checked against the extraction root *before* extraction proceeds. If any entry would land outside it, the entire upload is rejected with `UnsafeZipError` — no partial extraction of a malicious archive.
- **Zip-bomb limits**, all checked from the zip's central directory *before* extracting a single byte: total uncompressed size (`MAX_ZIP_UNCOMPRESSED_MB`), total entry count (`MAX_ZIP_ENTRIES` — guards against a "quantity bomb" of e.g. a million empty files, a resource cost that scales with file count regardless of byte size), and any single entry's compression ratio (`MAX_ZIP_COMPRESSION_RATIO` — the classic signal used by archives like "42.zip").
- **Raw upload size** (`MAX_UPLOAD_SIZE_MB`, API only) — enforced twice: a fast `Content-Length` pre-check that rejects an obviously oversized request immediately, then an authoritative streamed check (`_read_upload_capped`) that aborts as soon as the running byte total crosses the cap, so a client can't force an unbounded memory allocation just by claiming a small `Content-Length` and sending more anyway.

---

### 12. .gitignore-aware indexing

**Problem it solves:** without this, `node_modules/`, `dist/`, `coverage/`, vendored dependencies, and similar generated or vendored trees only get skipped if they happen to match the hardcoded `SKIP_DIRS` list in `config.py`. Anything project-specific — a custom build output directory, a `fixtures/` folder full of golden test data, a generated `protobuf/` tree — sails straight through, gets chunked, embedded, and dilutes retrieval with noise nobody wanted searched in the first place.

**How it works** (`utils/gitignore_filter.py`): reads the project's own root `.gitignore` (if present) and applies it with the same matching engine `git` itself uses (gitwildmatch syntax, via the [`pathspec`](https://github.com/cpburnz/python-pathspec) library) — so "what git tracks" and "what gets indexed" stay in sync automatically, with no project-specific config needed. Handles the "Download ZIP" shape from GitHub/GitLab, where the whole repo is wrapped in a single `reponame-branch/` folder, by also checking one level down for the `.gitignore`.

**Scope:** only the root `.gitignore` is honored — nested per-directory `.gitignore` files (e.g. a monorepo subproject with its own rules) are a known, deliberate limitation, since correctly resolving override precedence across directory levels is easy to get subtly wrong, and a single root file already covers the overwhelming majority of real repos. Toggle with `RESPECT_GITIGNORE` (default **on**).

---

### 13. Concurrent embedding

**Problem it solves:** embedding a large project one document at a time, sequentially, leaves most of the wait spent on network round-trips to Ollama rather than actual computation.

**How it works**: documents are embedded in batches of `EMBEDDING_BATCH_SIZE` (default 100) per Ollama call, with up to `MAX_EMBEDDING_WORKERS` (default 4) batches embedded concurrently — each an independent HTTP round trip with no shared state. Writes to Chroma stay single-threaded regardless, so concurrency only speeds up the embedding step, not the storage step. Set `MAX_EMBEDDING_WORKERS=1` to fall back to strictly sequential embedding (e.g. if your Ollama instance is CPU-bound and single-threaded, where concurrency mostly just adds scheduling overhead with little wall-clock benefit).

---

### 14. Observability: query telemetry & /metrics

**Problem it solves:** `qa_chain.py`'s existing structured log line is human-readable but not queryable without grepping/parsing log files by hand — answering "is retrieval slow, or coming back empty, more often than it used to?" meant doing that parsing yourself.

**How it works** (`core/query_log.py`, `GET /metrics`): every completed query appends one JSON Lines record — retrieval mode, hit count, retrieval/generation/total latency, estimated prompt/completion token counts, and the target collection — to a telemetry file (`QUERY_METRICS_LOG_PATH`, default `data/query_metrics.jsonl`). `GET /metrics` loads and aggregates that log into overall and per-mode summaries (avg/p50/p95 latency, hit-rate, avg token counts), optionally filtered by `since_minutes` or `collection`.

**Deliberately excluded:** the log never records the question text, the generated answer, or retrieved chunk content — only numeric/categorical telemetry. A codebase QA tool's queries can reference proprietary code and internal concerns; this log answers infrastructure questions, and is not meant to become a second, less-protected copy of every question anyone asked. Toggle with `ENABLE_QUERY_METRICS_LOG` (default **on**).

Writes are guarded by a lock (concurrent Streamlit sessions / API requests can log at the same time); reads tolerate a torn trailing line from an in-progress write.

---

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com)** installed and running locally
- Two Ollama models pulled:
  ```bash
  ollama pull qwen2.5:3b          # or any chat model you prefer
  ollama pull nomic-embed-text    # embedding model
  ```
- (Optional, for reranking) internet access on first run, to download the cross-encoder model from Hugging Face (~90MB, cached locally afterward)

---

## Installation

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd codebase-qa

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the env template and fill it in
cp .env.example .env
```

> `sentence-transformers` (for reranking) pulls in `torch` and is a fairly large install (~2GB). If you want a lighter install and don't need reranking, remove that line from `requirements.txt` before installing — the app detects its absence and falls back to unranked hybrid retrieval automatically.

---

## Configuration

Set these in `.env` (see `.env.example`):

| Variable | Required | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | Yes | Ollama server URL, e.g. `http://localhost:11434` |
| `OLLAMA_MODEL` | Yes | Chat model name, e.g. `qwen2.5:3b` |
| `OLLAMA_EMBEDDING_MODEL` | Yes | Embedding model name, e.g. `nomic-embed-text` |
| `CHROMA_DB_PATH` | Yes | Local path for the ChromaDB store, e.g. `./data/chroma_db` |
| `RETRIEVAL_MODE` | No | `vector` \| `mmr` \| `hybrid` \| `hybrid_rerank` (default: `hybrid_rerank`) |
| `LOG_LEVEL` | No | Python logging level (default: `INFO`) |
| `EMBEDDING_BATCH_SIZE` | No | Documents embedded per Ollama call (default: `100`) |
| `MAX_EMBEDDING_WORKERS` | No | Concurrent embedding batches; `1` = sequential (default: `4`) |
| `ENABLE_QUERY_REWRITE` | No | Conversation-aware standalone query rewriting (default: `true`) |
| `QUERY_REWRITE_HISTORY_TURNS` | No | Trailing chat turn-pairs considered when rewriting (default: `3`, set in code) |
| `ENABLE_QUERY_EXPANSION` | No | Multi-query expansion / RAG-Fusion — adds one LLM call per query (default: `false`) |
| `QUERY_EXPANSION_COUNT` | No | Paraphrases generated per query when expansion is on (default: `2`) |
| `RESPECT_GITIGNORE` | No | Exclude files matched by the project's root `.gitignore` during indexing (default: `true`) |
| `MAX_UPLOAD_SIZE_MB` | No | Max raw upload size, API only (default: `200`) |
| `MAX_ZIP_UNCOMPRESSED_MB` | No | Max total uncompressed size of a zip's contents (default: `500`) |
| `MAX_ZIP_ENTRIES` | No | Max number of entries in a zip (default: `20000`) |
| `MAX_ZIP_COMPRESSION_RATIO` | No | Max per-entry compression ratio before rejecting as a likely zip bomb (default: `100`) |
| `ENABLE_QUERY_METRICS_LOG` | No | Log per-query telemetry (latency, tokens, hit count) for `/metrics` (default: `true`) |
| `QUERY_METRICS_LOG_PATH` | No | Path to the query telemetry JSONL file (default: `data/query_metrics.jsonl`) |

A few retrieval/chunking constants are also tunable directly in `config.py` (not via env var): `CHUNK_SIZE` (1000), `CHUNK_OVERLAP` (150), `MAX_RETRIEVAL_DOCS` (6), `RETRIEVAL_FETCH_K` (20 — candidates pulled before fusion/MMR/rerank narrows to `MAX_RETRIEVAL_DOCS`), `MMR_LAMBDA` (0.5), `RRF_K` (60), and `RERANKER_MODEL`.

The app **fails fast** at startup with a clear error if any *required* variable is missing — it won't silently run with a broken config.

---

## Running the app

### Streamlit UI

```bash
streamlit run app/main.py
```
Opens at `http://localhost:8501`. Upload a project from the sidebar, wait for indexing to finish, then ask questions in the chat panel.

### FastAPI backend

```bash
uvicorn api:app --reload --app-dir app --port 8000
```
Interactive docs at `http://localhost:8000/docs`.

**Example: index a project**
```bash
curl -X POST http://localhost:8000/index \
  -F "project_name=my-project" \
  -F "file=@my-project.zip"
```

**Example: ask a question**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my-project",
    "question": "How does authentication work?",
    "retrieval_mode": "hybrid_rerank"
  }'
```

**Example: stream a question (SSE)**
```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"collection_name": "my-project", "question": "How does authentication work?"}'
```

**Example: check query telemetry for the last hour**
```bash
curl "http://localhost:8000/metrics?since_minutes=60"
```

### Docker

Runs Ollama, the Streamlit UI, and the FastAPI layer as separate containers, wired together with `docker-compose.yml`. Nothing needs to be installed on the host besides Docker itself.

```bash
cp .env.example .env   # fill in the same values as the manual setup above
docker compose up --build
```

- UI: `http://localhost:8501`
- API: `http://localhost:8000/docs`

The `ollama-init` service pulls `OLLAMA_MODEL` and `OLLAMA_EMBEDDING_MODEL` (from `.env`) into the `ollama` container on first startup, so the first query doesn't stall on an on-demand model pull. The Chroma DB and query metrics under `data/` persist on the host via a bind mount, and pulled Ollama models persist in a named volume — both survive `docker compose down` (add `-v` to also clear the Ollama models volume).

To run just one service against an Ollama already running on the host instead of the bundled container:
```bash
docker build -t codebase-qa .
docker run -p 8501:8501 --env-file .env \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v "$(pwd)/data:/app/data" codebase-qa
```
(swap the `streamlit run ...` default command for `uvicorn api:app --app-dir app --host 0.0.0.0 --port 8000` via `docker run ... codebase-qa uvicorn ...` to run the API instead.)

---

## Running the tests

```bash
pytest
```

341 tests, all pure/unit-level — no live Ollama or Chroma required (they run against mocked vector-store calls and hermetic env values, see `tests/conftest.py`). Covers: chunking (AST + heuristic), zip-slip and zip-bomb protection, `.gitignore` filtering, incremental sync diff logic, BM25 + RRF fusion (including multi-query), reranking (mocked model), retrieval filters (language/source/symbol), query rewriting and expansion (mocked LLM), query telemetry logging and aggregation, eval metrics, answer-quality keyword scoring, LLM-as-judge (mocked), and eval run-history logging.

```bash
pytest -v                           # verbose
pytest tests/test_ast_chunking.py   # a single file
```

---

## Running the evaluation harness

Requires a running Ollama instance (it performs real embedding + retrieval against a temporary `eval-codebase-qa-self` collection, indexing this project's own code; `--score-answers`/`--judge` also require it for generation).

```bash
python eval/run_eval.py                                   # all 4 modes, k=6
python eval/run_eval.py --modes hybrid hybrid_rerank       # just these two
python eval/run_eval.py --k 3                              # different k
python eval/run_eval.py --score-answers                    # + full generation + keyword scoring
python eval/run_eval.py --score-answers --judge             # + LLM-as-judge grading
python eval/run_eval.py --history                           # print the last 10 logged runs, don't run a new one
python eval/run_eval.py --history 30                        # print the last 30
python eval/run_eval.py --no-log                            # run without appending to the log
python eval/run_eval.py --project-dir /path/to/other/repo --collection other-eval --reindex
```

---

## Project structure

```
codebase-qa/
├── app/
│   ├── main.py                    # Streamlit entry point
│   ├── api.py                     # FastAPI entry point (incl. /metrics, /query/stream)
│   ├── config.py                  # env config, fail-fast validation
│   ├── components/
│   │   ├── sidebar.py             # upload + indexing UI, collection/source/symbol pickers
│   │   ├── chat.py                # chat UI, streams answers
│   │   └── file_tree.py           # indexed-files tree view
│   ├── core/
│   │   ├── ingestion.py           # file -> chunks -> Document, content hashing
│   │   ├── ast_chunking.py        # tree-sitter AST chunking, symbol metadata
│   │   ├── indexing.py            # incremental sync (diff + upsert + deletion)
│   │   ├── vectorstore.py         # Chroma client, collections, CRUD
│   │   ├── retriever.py           # unified retrieval pipeline (4 modes + filters)
│   │   ├── hybrid_retriever.py    # BM25 + vector + RRF, incl. multi-query fusion
│   │   ├── reranker.py            # cross-encoder reranking
│   │   ├── query_rewrite.py       # standalone rewriting + multi-query expansion
│   │   ├── query_log.py           # per-query JSONL telemetry + aggregation
│   │   └── qa_chain.py            # prompt assembly, LLM streaming, metrics
│   └── utils/
│       ├── file_handler.py        # zip/dir extraction, zip-slip + zip-bomb guards
│       ├── gitignore_filter.py    # .gitignore-aware indexing filter
│       └── language_detect.py     # extension -> language mapping
├── eval/
│   ├── eval_set.py                # 21 hand-built Q&A ground-truth pairs
│   ├── metrics.py                 # precision@k, recall@k, MRR
│   ├── answer_metrics.py          # keyword coverage / hit scoring for generated answers
│   ├── llm_judge.py               # LLM-as-judge answer grading (opt-in)
│   ├── eval_log.py                # JSONL run history + config snapshot + diffing
│   └── run_eval.py                # CLI harness
├── tests/                         # 341 pytest unit tests
├── requirements.txt
├── pytest.ini
├── .env.example
└── README.md
```

---

## Supported languages

| Capability | Languages |
|---|---|
| **AST-based chunking** (whole functions/classes, symbol metadata) | Python, JavaScript, JSX, TypeScript, TSX, Java, Go, Rust, C, C++ |
| **Indexed** (heuristic chunking fallback) | Everything in `SUPPORTED_EXTENSIONS` in `config.py` — also includes Ruby, PHP, Swift, Kotlin, Scala, C#, HTML/CSS/SCSS, JSON/YAML/TOML, Markdown, shell, SQL, R, Lua |

---

## Known limitations

- **Single-machine, single-user** — no auth, no multi-tenant isolation; this is a local developer tool, not a hosted service.
- **BM25 index is rebuilt per query** from whatever's currently in the collection — fine at the scale of one codebase (hundreds–low thousands of chunks), not optimized for very large corpora.
- **Reranker needs network on first use** to download model weights from Hugging Face; after that, it's fully local.
- **Query rewriting/expansion adds LLM round trips** — rewriting only fires on real follow-ups (cheap), but expansion runs on every query when enabled, adding one full extra generation call to each question's latency.
- **`.gitignore` support is root-only** — nested per-directory `.gitignore` files in a monorepo aren't resolved; only the project's top-level `.gitignore` is honored.
- **Symbol-name filtering only covers AST-chunked languages** — chunks produced by the heuristic fallback splitter carry no `symbol_name`, so a symbol filter silently excludes them.
- **No containerization or CI pipeline yet** — everything currently runs from a local Python environment.

---

## Troubleshooting

**App won't start / "Missing required environment variable"**
`.env` is missing a required value — compare it against `.env.example`.

**"No relevant code found" on every query**
Check `ollama list` shows your configured models are pulled, and that `OLLAMA_BASE_URL` is reachable (`curl http://localhost:11434`).

**Reranking silently isn't happening**
Check logs for `sentence-transformers not installed` or a model-load failure — reranking fails soft, so a query still succeeds, just without that stage. Confirm you have network access for the first run (subsequent runs use the cached model).

**Zip upload rejected as "unsafe"**
The archive contains an entry that would extract outside the target directory (a zip-slip attempt, or occasionally a mis-packed archive using absolute paths) — re-zip the project so all paths are relative.

**Zip upload rejected as too large / too many entries / suspicious compression ratio**
The archive tripped one of the zip-bomb limits (`MAX_ZIP_UNCOMPRESSED_MB`, `MAX_ZIP_ENTRIES`, `MAX_ZIP_COMPRESSION_RATIO`). If it's a legitimate large project, raise the relevant limit in `.env`.

**A file I expected to be indexed is missing**
Check whether it's matched by your project's root `.gitignore` — indexing respects it by default (`RESPECT_GITIGNORE=true`). Also check `SKIP_DIRS` / `SKIP_EXTENSIONS` / `SUPPORTED_EXTENSIONS` in `config.py`.

**`/metrics` returns all zeros / empty**
`ENABLE_QUERY_METRICS_LOG` may be off, or no queries have been logged yet at `QUERY_METRICS_LOG_PATH`. This aggregates real traffic through the app/API, not the eval harness — see `eval/run_eval.py --history` for eval-run history instead.