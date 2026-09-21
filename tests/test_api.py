
"""
Tests for the FastAPI layer's SSE streaming query endpoint (/query/stream)
and the low-level SSE framing helpers it's built on.

These use FastAPI's TestClient rather than a live server, and mock
run_qa_streaming / collection_exists so nothing here touches Ollama or a
real Chroma collection.
"""
import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app, _sse_event


client = TestClient(app)


def _parse_sse_stream(raw_text: str):
    """
    Parses raw SSE response text into a list of (event_type, data_dict)
    tuples, in order — mirrors what a real SSE client would receive.
    """
    events = []
    for block in raw_text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_type = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event_type = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        events.append((event_type, data))
    return events


def _fake_qa_events(*, tokens=("Hello", " world"), sources=None, metrics=None, error=None):
    """A stand-in generator matching run_qa_streaming's event shape."""
    if error is not None:
        yield {"type": "error", "message": error}
        return
    for t in tokens:
        yield {"type": "token", "value": t}
    yield {"type": "sources", "value": sources or []}
    yield {"type": "metrics", "value": metrics or {"hit_count": 1}}


# ---------- _sse_event (framing helper) ----------

def test_sse_event_format():
    frame = _sse_event("token", {"value": "hi"})
    assert frame == 'event: token\ndata: {"value": "hi"}\n\n'


def test_sse_event_json_encodes_embedded_newlines_safely():
    # A raw newline in the payload must not create a spurious blank-line
    # frame boundary — JSON-encoding escapes it to \n within the string.
    frame = _sse_event("token", {"value": "line one\nline two"})
    assert frame.count("\n\n") == 1  # exactly one true frame terminator
    assert frame.endswith("\n\n")


# ---------- /query/stream ----------

def test_stream_404_before_streaming_starts_when_collection_missing():
    with patch("api.collection_exists", return_value=False):
        response = client.post(
            "/query/stream",
            json={"collection_name": "nonexistent", "question": "how does X work?"},
        )
    assert response.status_code == 404


def test_stream_emits_token_sources_metrics_then_done_in_order():
    with patch("api.collection_exists", return_value=True), \
         patch("api.run_qa_streaming", return_value=_fake_qa_events(
             tokens=("Hello", " world"),
             sources=[{"source": "a.py"}],
             metrics={"hit_count": 3},
         )):
        response = client.post(
            "/query/stream",
            json={"collection_name": "proj", "question": "how does X work?"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_stream(response.text)
    types = [e[0] for e in events]
    assert types == ["token", "token", "sources", "metrics", "done"]
    assert events[0][1] == {"value": "Hello"}
    assert events[1][1] == {"value": " world"}
    assert events[2][1] == {"value": [{"source": "a.py"}]}
    assert events[3][1] == {"value": {"hit_count": 3}}
    assert events[4][1] == {}


def test_stream_emits_error_event_and_still_ends_with_done():
    with patch("api.collection_exists", return_value=True), \
         patch("api.run_qa_streaming", return_value=_fake_qa_events(error="No relevant code found.")):
        response = client.post(
            "/query/stream",
            json={"collection_name": "proj", "question": "how does X work?"},
        )

    events = _parse_sse_stream(response.text)
    assert [e[0] for e in events] == ["error", "done"]
    assert events[0][1] == {"message": "No relevant code found."}


def test_stream_still_returns_200_and_in_band_error_on_exception_mid_stream():
    def _raising_generator(**kwargs):
        yield {"type": "token", "value": "partial answer..."}
        raise RuntimeError("ollama connection dropped")

    with patch("api.collection_exists", return_value=True), \
         patch("api.run_qa_streaming", side_effect=lambda **kw: _raising_generator(**kw)):
        response = client.post(
            "/query/stream",
            json={"collection_name": "proj", "question": "how does X work?"},
        )

    # status is already committed by the time the exception happens mid-stream
    assert response.status_code == 200
    events = _parse_sse_stream(response.text)
    types = [e[0] for e in events]
    assert types == ["token", "error", "done"]
    assert "ollama connection dropped" in events[1][1]["message"]


def test_stream_passes_through_filters_and_chat_history():
    with patch("api.collection_exists", return_value=True), \
         patch("api.run_qa_streaming", return_value=_fake_qa_events()) as mock_run:
        client.post(
            "/query/stream",
            json={
                "collection_name": "proj",
                "question": "how does X work?",
                "filter_language": "Python",
                "filter_source": "app/main.py",
                "filter_symbol": "retrieve_documents",
                "chat_history": [{"role": "user", "content": "earlier question"}],
            },
        )

    kwargs = mock_run.call_args.kwargs
    assert kwargs["filter_language"] == "Python"
    assert kwargs["filter_source"] == "app/main.py"
    assert kwargs["filter_symbol"] == "retrieve_documents"
    assert kwargs["chat_history"] == [{"role": "user", "content": "earlier question"}]


def test_stream_headers_disable_proxy_buffering():
    with patch("api.collection_exists", return_value=True), \
         patch("api.run_qa_streaming", return_value=_fake_qa_events()):
        response = client.post(
            "/query/stream",
            json={"collection_name": "proj", "question": "how does X work?"},
        )

    assert response.headers.get("cache-control") == "no-cache"
    assert response.headers.get("x-accel-buffering") == "no"


def test_stream_and_non_streaming_query_produce_consistent_answer_text():
    # /query/stream's token events, concatenated, should equal /query's
    # single "answer" field for the same underlying event sequence — the
    # two endpoints are re-presentations of the same run_qa_streaming
    # output, not divergent logic.
    full_metrics = {
        "hit_count": 1,
        "retrieval_latency_ms": 1.0,
        "generation_latency_ms": 1.0,
        "total_latency_ms": 2.0,
        "prompt_tokens_estimate": 10,
        "completion_tokens_estimate": 5,
    }
    events = list(_fake_qa_events(tokens=("The ", "answer ", "is ", "42."), metrics=full_metrics))

    with patch("api.collection_exists", return_value=True), \
         patch("api.run_qa_streaming", return_value=iter(events)):
        stream_response = client.post(
            "/query/stream",
            json={"collection_name": "proj", "question": "q"},
        )

    with patch("api.collection_exists", return_value=True), \
         patch("api.run_qa_streaming", return_value=iter(events)):
        plain_response = client.post(
            "/query",
            json={"collection_name": "proj", "question": "q"},
        )

    stream_answer = "".join(
        e[1]["value"] for e in _parse_sse_stream(stream_response.text) if e[0] == "token"
    )
    assert stream_answer == plain_response.json()["answer"] == "The answer is 42."


# ---------- _read_upload_capped ----------

import asyncio
import io
import os
import zipfile

import pytest
from fastapi import HTTPException

from api import _read_upload_capped


class _FakeUploadFile:
    """Minimal stand-in for fastapi.UploadFile: async .read(size) over in-memory bytes."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    async def read(self, size: int) -> bytes:
        chunk = self._data[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk


def test_read_upload_capped_returns_full_content_within_limit():
    data = b"x" * 5000
    result = asyncio.run(_read_upload_capped(_FakeUploadFile(data), max_bytes=10_000))
    assert result == data


def test_read_upload_capped_raises_413_when_exceeding_limit():
    data = b"x" * 20_000
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_read_upload_capped(_FakeUploadFile(data), max_bytes=10_000))
    assert exc_info.value.status_code == 413


def test_read_upload_capped_stops_reading_before_consuming_entire_body():
    # chunk_size inside _read_upload_capped is fixed at 1MB; 3MB of data
    # with a 1.5MB cap should bail after the 2nd 1MB chunk (running total
    # 2MB > 1.5MB), never reading (or buffering) the 3rd.
    read_sizes = []

    class _TrackingUploadFile(_FakeUploadFile):
        async def read(self, size: int) -> bytes:
            read_sizes.append(size)
            return await super().read(size)

    data = b"x" * (3 * 1024 * 1024)
    with pytest.raises(HTTPException):
        asyncio.run(_read_upload_capped(_TrackingUploadFile(data), max_bytes=int(1.5 * 1024 * 1024)))

    assert len(read_sizes) == 2


def test_read_upload_capped_empty_file_returns_empty_bytes():
    result = asyncio.run(_read_upload_capped(_FakeUploadFile(b""), max_bytes=10_000))
    assert result == b""


# ---------- /index upload size + zip safety (end-to-end through the endpoint) ----------

def _make_zip_bytes(entries: dict, compression=zipfile.ZIP_DEFLATED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression) as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_index_rejects_upload_over_max_size(monkeypatch):
    import api as api_module
    monkeypatch.setattr(api_module, "MAX_UPLOAD_SIZE_MB", 1)

    # Incompressible content + ZIP_STORED (no compression) so the actual
    # uploaded .zip bytes exceed the 1MB cap, not just its uncompressed content.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("random.bin", os.urandom(1_200_000))
    zip_bytes = buf.getvalue()

    response = client.post(
        "/index",
        data={"project_name": "test"},
        files={"file": ("big.zip", zip_bytes, "application/zip")},
    )
    assert response.status_code == 413


def test_index_accepts_upload_within_size_limit():
    zip_bytes = _make_zip_bytes({"main.py": "print('hello')"})
    fake_stats = {
        "new_files": 1, "updated_files": 0, "skipped_files": 0, "deleted_files": 0,
        "new_file_paths": ["main.py"], "updated_file_paths": [], "skipped_file_paths": [],
        "deleted_file_paths": [], "chunks_embedded": 1, "collection_chunk_count": 1,
        "chunk_methods": {"heuristic": 1}, "elapsed_ms": 5.0,
    }

    with patch("api.sync_files_to_collection", return_value=fake_stats):
        response = client.post(
            "/index",
            data={"project_name": "test-proj"},
            files={"file": ("small.zip", zip_bytes, "application/zip")},
        )

    assert response.status_code == 200
    assert response.json()["new_files"] == 1


def test_index_returns_400_for_corrupted_zip():
    response = client.post(
        "/index",
        data={"project_name": "test"},
        files={"file": ("bad.zip", b"this is not a real zip file", "application/zip")},
    )
    assert response.status_code == 400
    assert "valid zip" in response.json()["detail"].lower()


def test_index_returns_400_for_zip_bomb_shaped_archive(monkeypatch):
    import utils.file_handler as fh
    monkeypatch.setattr(fh, "MAX_ZIP_UNCOMPRESSED_MB", 1)

    zip_bytes = _make_zip_bytes({"big.py": "x" * (2 * 1024 * 1024)})

    response = client.post(
        "/index",
        data={"project_name": "test"},
        files={"file": ("bomb.zip", zip_bytes, "application/zip")},
    )
    assert response.status_code == 400
    assert "uncompressed" in response.json()["detail"].lower()


def test_index_returns_400_for_excessive_entry_count(monkeypatch):
    import utils.file_handler as fh
    monkeypatch.setattr(fh, "MAX_ZIP_ENTRIES", 5)

    zip_bytes = _make_zip_bytes({f"file_{i}.py": "x = 1" for i in range(10)})

    response = client.post(
        "/index",
        data={"project_name": "test"},
        files={"file": ("many.zip", zip_bytes, "application/zip")},
    )
    assert response.status_code == 400
    assert "entries" in response.json()["detail"].lower()

# ---------- /metrics ----------

def _fake_records():
    return [
        {
            "timestamp": "2026-06-01T00:00:00+00:00", "collection": "proj",
            "retrieval_mode": "vector", "hit_count": 1, "error": False,
            "retrieval_latency_ms": 50.0, "generation_latency_ms": 200.0, "total_latency_ms": 250.0,
            "prompt_tokens_estimate": 100, "completion_tokens_estimate": 30,
        },
        {
            "timestamp": "2026-06-01T00:01:00+00:00", "collection": "proj",
            "retrieval_mode": "hybrid_rerank", "hit_count": 0, "error": True,
            "retrieval_latency_ms": 30.0, "generation_latency_ms": 0.0, "total_latency_ms": 30.0,
            "prompt_tokens_estimate": 0, "completion_tokens_estimate": 0,
        },
        {
            "timestamp": "2026-06-01T00:02:00+00:00", "collection": "other-proj",
            "retrieval_mode": "vector", "hit_count": 2, "error": False,
            "retrieval_latency_ms": 40.0, "generation_latency_ms": 300.0, "total_latency_ms": 340.0,
            "prompt_tokens_estimate": 150, "completion_tokens_estimate": 60,
        },
    ]


def test_metrics_returns_overall_and_by_mode_breakdown():
    with patch("api.load_query_records", return_value=_fake_records()):
        response = client.get("/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["overall"]["count"] == 3
    assert body["overall"]["error_count"] == 1
    assert set(body["by_mode"].keys()) == {"vector", "hybrid_rerank"}
    assert body["by_mode"]["vector"]["count"] == 2


def test_metrics_empty_log_returns_zeroed_response():
    with patch("api.load_query_records", return_value=[]):
        response = client.get("/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["overall"]["count"] == 0
    assert body["by_mode"] == {}


def test_metrics_filters_by_collection():
    with patch("api.load_query_records", return_value=_fake_records()):
        response = client.get("/metrics", params={"collection": "other-proj"})

    body = response.json()
    assert body["overall"]["count"] == 1
    assert body["collection_filter"] == "other-proj"


def test_metrics_since_minutes_passed_through_as_datetime():
    with patch("api.load_query_records", return_value=[]) as mock_load:
        response = client.get("/metrics", params={"since_minutes": 60})

    assert response.status_code == 200
    called_since = mock_load.call_args.kwargs.get("since")
    assert called_since is not None
    body = response.json()
    assert body["since"] is not None


def test_metrics_no_since_minutes_means_since_is_none():
    with patch("api.load_query_records", return_value=[]) as mock_load:
        client.get("/metrics")

    assert mock_load.call_args.kwargs.get("since") is None


def test_metrics_rejects_non_positive_since_minutes():
    response = client.get("/metrics", params={"since_minutes": 0})
    assert response.status_code == 422

    response = client.get("/metrics", params={"since_minutes": -5})
    assert response.status_code == 422


def test_metrics_uses_configured_log_path():
    with patch("api.load_query_records", return_value=[]) as mock_load, \
         patch("api.QUERY_METRICS_LOG_PATH", "/tmp/some-configured-path.jsonl"):
        client.get("/metrics")

    assert mock_load.call_args.args[0] == "/tmp/some-configured-path.jsonl"