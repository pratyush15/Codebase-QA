
"""
FastAPI layer for Codebase Q&A.

This exposes the same core/ logic (ingestion, indexing, retrieval, qa_chain)
that the Streamlit UI uses, but as a stateless HTTP API — nothing here reads
or writes st.session_state. The Streamlit app in main.py can keep using
core/ directly, or be pointed at this API instead; either way, indexing and
querying no longer require a Streamlit session to work (e.g. can be called
from a CI job, a CLI, or another service).

Run with:
    uvicorn api:app --reload --app-dir app --port 8000
(from the project root, so `app/` is the working dir the imports expect —
mirrors how main.py is run today via `streamlit run app/main.py`.)
"""
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Generator, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from config import MAX_UPLOAD_SIZE_MB, QUERY_METRICS_LOG_PATH
from core.qa_chain import check_ollama_connection, run_qa_streaming
from core.indexing import sync_files_to_collection
from core.query_log import load_query_records, aggregate_query_metrics
from core.vectorstore import (
    list_collections,
    delete_collection,
    collection_exists,
    sanitize_collection_name,
    get_collection_sources,
)
from utils.file_handler import extract_files_from_zip, get_summary, UnsafeZipError

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Codebase Q&A API",
    description="Index codebases and query them with a local LLM over your own vector store.",
    version="1.0.0",
)


# ---------- Schemas ----------

class CollectionInfo(BaseModel):
    name: str
    chunk_count: int


class IndexResponse(BaseModel):
    collection: str
    new_files: int
    updated_files: int
    skipped_files: int
    deleted_files: int
    chunks_embedded: int
    collection_chunk_count: int
    chunk_methods: dict
    elapsed_ms: float
    total_size_kb: float


class ChatTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class QueryRequest(BaseModel):
    collection_name: str
    question: str
    filter_language: Optional[str] = None
    filter_source: Optional[str] = None
    filter_symbol: Optional[str] = None  # exact match against a chunk's symbol_name (function/class/method it defines)
    retrieval_mode: Optional[str] = None  # "vector" | "mmr" | "hybrid" | "hybrid_rerank"; defaults to config.RETRIEVAL_MODE
    chat_history: Optional[List[ChatTurn]] = None  # prior turns, oldest first; do NOT include `question` itself


class SourceInfo(BaseModel):
    source: str
    language: str = ""
    role: str = ""
    chunk_index: int = 0
    total_chunks: int = 1
    symbol_name: str = ""
    symbol_type: str = ""
    parent_symbol: str = ""


class QueryMetrics(BaseModel):
    retrieval_mode: str = ""
    hit_count: int
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    prompt_tokens_estimate: int
    completion_tokens_estimate: int
    query_rewritten: bool = False
    rewritten_query: Optional[str] = None
    query_rewrite_latency_ms: float = 0.0
    expanded_queries: Optional[List[str]] = None
    query_expansion_latency_ms: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    metrics: QueryMetrics


class MetricsSummary(BaseModel):
    count: int
    error_count: int
    hit_rate: float
    avg_total_latency_ms: float
    p50_total_latency_ms: float
    p95_total_latency_ms: float
    avg_retrieval_latency_ms: float
    avg_generation_latency_ms: float
    avg_prompt_tokens: float
    avg_completion_tokens: float


class MetricsResponse(BaseModel):
    overall: MetricsSummary
    by_mode: Dict[str, MetricsSummary]
    since: Optional[str] = None
    collection_filter: Optional[str] = None


# ---------- Routes ----------

@app.get("/health")
def health():
    result = check_ollama_connection()
    if result["status"] != "ok":
        return JSONResponse(status_code=503, content=result)
    return result


@app.get("/collections", response_model=List[CollectionInfo])
def get_collections():
    return list_collections()


@app.delete("/collections/{name}")
def remove_collection(name: str):
    if not collection_exists(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")
    if not delete_collection(name):
        raise HTTPException(status_code=500, detail=f"Failed to delete collection '{name}'.")
    return {"deleted": name}


@app.get("/collections/{name}/sources", response_model=List[str])
def get_sources(name: str):
    if not collection_exists(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")
    return get_collection_sources(name)


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics(since_minutes: Optional[int] = None, collection: Optional[str] = None):
    """
    Aggregated query telemetry — avg/p50/p95 latency, token estimates, and
    hit-rate, both overall and broken down by retrieval mode. Backed by
    the JSONL log core.query_log writes to after every completed query
    (see QUERY_METRICS_LOG_PATH in config.py); this endpoint just loads
    and aggregates that log, so the numbers reflect actual query traffic
    through this API and/or the Streamlit UI — a lightweight alternative
    to grepping qa_chain.py's logger.info() lines by hand. (For synthetic
    eval-set metrics instead of live traffic, see eval/run_eval.py.)

    since_minutes: only include queries from the last N minutes (omit for
    the entire logged history).
    collection: only include queries against this collection (omit to
    combine every collection's traffic together).
    """
    since = None
    if since_minutes is not None:
        if since_minutes <= 0:
            raise HTTPException(status_code=422, detail="since_minutes must be a positive integer.")
        since = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)

    records = load_query_records(QUERY_METRICS_LOG_PATH, since=since)
    if collection:
        records = [r for r in records if r.get("collection") == collection]

    aggregated = aggregate_query_metrics(records)

    return MetricsResponse(
        overall=aggregated["overall"],
        by_mode=aggregated["by_mode"],
        since=since.isoformat() if since else None,
        collection_filter=collection,
    )


async def _read_upload_capped(file: UploadFile, max_bytes: int) -> bytes:
    """
    Reads an UploadFile in fixed-size chunks, aborting as soon as the
    running total exceeds max_bytes — rather than `await file.read()`,
    which pulls the *entire* upload into one Python bytes object before
    any size check is possible. A client sending a huge (or infinite)
    body can't force that allocation past the configured cap.
    """
    chunk_size = 1024 * 1024  # 1MB
    chunks: List[bytes] = []
    total = 0

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds the {max_bytes // (1024 * 1024)}MB limit.",
            )
        chunks.append(chunk)

    return b"".join(chunks)


@app.post("/index", response_model=IndexResponse)
async def index_codebase(
    request: Request,
    project_name: str = Form(..., description="Human-readable project name; sanitized into a collection name."),
    file: UploadFile = File(..., description="A .zip archive of the codebase to index."),
    full_sync: bool = Form(
        True,
        description=(
            "Whether this zip represents the entire current state of the project. "
            "When true (the default), any previously-indexed file missing from this "
            "upload is treated as deleted and its chunks are removed. Set false if "
            "this zip is a deliberate partial/subset upload."
        ),
    ),
):
    if not project_name.strip():
        raise HTTPException(status_code=422, detail="project_name must not be empty.")

    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=422, detail="Only .zip uploads are supported by this endpoint.")

    # Fast pre-check against the request's declared Content-Length, before
    # reading any of the body: catches an obviously-oversized upload
    # immediately. Not authoritative on its own — Content-Length can be
    # absent or (in principle) misreported — so _read_upload_capped below
    # is the real enforcement, checked as bytes actually arrive.
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds the {MAX_UPLOAD_SIZE_MB}MB limit.",
                )
        except ValueError:
            pass  # malformed header — fall through to the authoritative streamed check

    zip_bytes = await _read_upload_capped(file, MAX_UPLOAD_SIZE_MB * 1024 * 1024)

    try:
        files = extract_files_from_zip(zip_bytes)
    except UnsafeZipError as e:
        logger.warning("Rejected unsafe zip upload for project '%s': %s", project_name, e)
        raise HTTPException(status_code=400, detail=str(e))

    if not files:
        raise HTTPException(status_code=422, detail="No readable code files found in the archive.")

    collection_name = sanitize_collection_name(project_name.strip())
    summary = get_summary(files)

    try:
        stats = sync_files_to_collection(collection_name, files, full_sync=full_sync)
    except Exception as e:
        logger.exception("Indexing failed for collection '%s'", collection_name)
        raise HTTPException(
            status_code=503,
            detail=f"Indexing failed — is Ollama running and reachable? ({e})",
        )

    return IndexResponse(
        collection=collection_name,
        total_size_kb=summary["total_size_kb"],
        **stats,
    )


@app.post("/query", response_model=QueryResponse)
def query_codebase(req: QueryRequest):
    if not collection_exists(req.collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{req.collection_name}' not found.")

    answer_parts: List[str] = []
    sources: List[dict] = []
    metrics: Optional[dict] = None
    error_message: Optional[str] = None

    try:
        for event in run_qa_streaming(
            collection_name=req.collection_name,
            question=req.question,
            filter_language=req.filter_language,
            filter_source=req.filter_source,
            filter_symbol=req.filter_symbol,
            chat_history=[turn.model_dump() for turn in req.chat_history] if req.chat_history else None,
            **({"retrieval_mode": req.retrieval_mode} if req.retrieval_mode else {}),
        ):
            if event["type"] == "token":
                answer_parts.append(event["value"])
            elif event["type"] == "sources":
                sources = event["value"]
            elif event["type"] == "metrics":
                metrics = event["value"]
            elif event["type"] == "error":
                error_message = event["message"]
    except Exception as e:
        logger.exception("Query failed for collection '%s'", req.collection_name)
        raise HTTPException(
            status_code=503,
            detail=f"Query failed — is Ollama running and reachable? ({e})",
        )

    if error_message:
        raise HTTPException(status_code=404, detail=error_message)

    return QueryResponse(
        answer="".join(answer_parts),
        sources=sources,
        metrics=metrics,
    )


def _sse_event(event_type: str, data: dict) -> str:
    """
    Formats one Server-Sent Event. The payload is always JSON-encoded —
    even the plain-string "token" events — because an SSE message ends at
    the first blank line; a raw token containing an embedded newline
    (extremely common mid-generation) would otherwise silently truncate
    the frame. JSON-encoding sidesteps that entirely.
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _stream_qa_events(req: QueryRequest) -> Generator[str, None, None]:
    """
    Re-emits run_qa_streaming's internal events as SSE frames, 1:1 by
    type ("token" / "sources" / "metrics" / "error"), then always emits a
    final "done" frame so clients have one unambiguous signal to stop
    listening — including after an in-band "error" event (e.g. "no
    relevant code found"), which is a normal, expected end state here,
    not a connection failure.

    Unlike /query's collection-existence check, a failure that happens
    *during* iteration (e.g. Ollama drops mid-answer) can't change the
    HTTP status code — the 200 response line and headers already went out
    the moment the client received the first byte of the stream — so it's
    surfaced as an in-band "error" event instead, the same way a real SSE
    client is expected to handle a mid-stream failure from any provider.
    """
    try:
        for event in run_qa_streaming(
            collection_name=req.collection_name,
            question=req.question,
            filter_language=req.filter_language,
            filter_source=req.filter_source,
            filter_symbol=req.filter_symbol,
            chat_history=[turn.model_dump() for turn in req.chat_history] if req.chat_history else None,
            **({"retrieval_mode": req.retrieval_mode} if req.retrieval_mode else {}),
        ):
            if event["type"] == "token":
                yield _sse_event("token", {"value": event["value"]})
            elif event["type"] == "sources":
                yield _sse_event("sources", {"value": event["value"]})
            elif event["type"] == "metrics":
                yield _sse_event("metrics", {"value": event["value"]})
            elif event["type"] == "error":
                yield _sse_event("error", {"message": event["message"]})
    except Exception as e:
        logger.exception("Streaming query failed for collection '%s'", req.collection_name)
        yield _sse_event("error", {"message": f"Query failed — is Ollama running and reachable? ({e})"})

    yield _sse_event("done", {})


@app.post("/query/stream")
def query_codebase_stream(req: QueryRequest):
    """
    Same request shape as /query, but responds with a Server-Sent Events
    stream instead of one buffered JSON blob — the API-consumer equivalent
    of the Streamlit chat UI's token-by-token rendering, instead of
    waiting for the full answer before showing anything.

    Event types, one JSON-encoded payload per "data:" line:
      - "token"   {"value": "<partial answer text>"}   — zero or more, in order
      - "sources" {"value": [SourceInfo, ...]}          — at most one, after the last token
      - "metrics" {"value": QueryMetrics}               — at most one, after "sources"
      - "error"   {"message": "<human-readable message>"} — at most one; if present,
                   no further "token"/"sources"/"metrics" events follow it
      - "done"    {}                                     — always exactly one, last event of the stream

    A minimal JS client:
        const es = new EventSource(...); // or fetch()+ReadableStream for POST
        es.addEventListener("token", (e) => append(JSON.parse(e.data).value));
        es.addEventListener("done", () => es.close());

    Collection-not-found is still a normal 404, raised before the stream
    starts — only failures that happen *during* generation are reported as
    an in-band "error" event (see _stream_qa_events for why).
    """
    if not collection_exists(req.collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{req.collection_name}' not found.")

    return StreamingResponse(
        _stream_qa_events(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Some reverse proxies (nginx) buffer proxied responses by
            # default, which defeats the point of a token-by-token stream.
            # Harmless if nothing's proxying this.
            "X-Accel-Buffering": "no",
        },
    )