"""
Lightweight, structured log of every completed query's telemetry (latency,
token estimates, retrieval mode, hit count) — separate from the existing
`logger.info(...)` line in qa_chain.py, which is human-readable but not
queryable without grepping/parsing log files by hand.

This is what /metrics (api.py) aggregates over. Same JSONL-append
pattern as eval/eval_log.py, for the same reasons: zero new dependencies,
one record per line, trivially appendable/readable, no schema migration
story to worry about.

Deliberately does NOT record the question text, the generated answer, or
retrieved chunk content — only numeric/categorical telemetry (latency,
token counts, mode, hit count, collection name). A codebase QA tool's
queries can reference proprietary code and internal concerns; this log is
meant to answer "is retrieval slow, or coming back empty, more often than
it used to?" — not to become a second, less-protected copy of every
question anyone ever asked the tool.
"""
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Guards writes only (not reads) — record_query() can be called from
# concurrent requests (Streamlit sessions, API requests) and JSONL append
# is not inherently atomic across threads on every platform; a lock keeps
# concurrent writers from interleaving mid-line and corrupting the file.
# Reads (load_query_records) don't need it: a reader either sees a line or
# doesn't, and a torn trailing line from an in-progress write is already
# handled by load_query_records' malformed-line tolerance.
_write_lock = threading.Lock()

# Fields recorded per query. Kept explicit (rather than "whatever kwargs
# happen to be passed") so it's obvious at a glance exactly what this log
# does and doesn't capture — see the module docstring for why question/
# answer text is deliberately excluded.
_RECORD_FIELDS = (
    "collection",
    "retrieval_mode",
    "hit_count",
    "retrieval_latency_ms",
    "generation_latency_ms",
    "total_latency_ms",
    "prompt_tokens_estimate",
    "completion_tokens_estimate",
    "query_rewritten",
    "query_expansion_count",
    "error",
)


def record_query(
    log_path,
    *,
    collection: str,
    retrieval_mode: str,
    hit_count: int = 0,
    retrieval_latency_ms: float = 0.0,
    generation_latency_ms: float = 0.0,
    total_latency_ms: float = 0.0,
    prompt_tokens_estimate: int = 0,
    completion_tokens_estimate: int = 0,
    query_rewritten: bool = False,
    query_expansion_count: int = 0,
    error: bool = False,
) -> None:
    """
    Appends one query's telemetry as a JSON line.

    error=True records a query that found no relevant chunks at all (the
    "no results" path in qa_chain.run_qa_streaming) — recording it, rather
    than skipping metrics entirely for that path, is what makes hit-rate
    and error-rate actually computable from this log; a log that only ever
    contains successful queries can't tell you how often queries fail.

    Never raises: a metrics-logging problem (disk full, permissions,
    anything) is logged and swallowed here rather than propagated — a
    telemetry write must never be able to break an actual query response
    reaching the user.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collection": collection,
        "retrieval_mode": retrieval_mode,
        "hit_count": hit_count,
        "retrieval_latency_ms": retrieval_latency_ms,
        "generation_latency_ms": generation_latency_ms,
        "total_latency_ms": total_latency_ms,
        "prompt_tokens_estimate": prompt_tokens_estimate,
        "completion_tokens_estimate": completion_tokens_estimate,
        "query_rewritten": query_rewritten,
        "query_expansion_count": query_expansion_count,
        "error": error,
    }

    try:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
    except Exception:
        logger.exception("Failed to write query metrics record to %s", log_path)


def load_query_records(log_path, since: Optional[datetime] = None) -> List[Dict]:
    """
    Reads every query record from the log, oldest first, optionally
    filtered to timestamp >= since (since must be timezone-aware, to match
    the timezone-aware timestamps record_query writes).

    A missing file returns an empty list (no queries logged yet is not an
    error). A malformed or unparseable-timestamp line is skipped rather
    than aborting the whole load — one corrupted line (e.g. from a write
    that lost a race with a process crash) shouldn't make the rest of the
    log unreadable.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return []

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed query metrics line %d in %s", line_no, log_path)
                continue

            if since is not None:
                try:
                    ts = datetime.fromisoformat(record["timestamp"])
                except (KeyError, TypeError, ValueError):
                    continue
                if ts < since:
                    continue

            records.append(record)

    return records


def _avg(values: List[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _percentile(values: List[float], pct: float) -> float:
    """
    Nearest-rank percentile over `values` (0.0 <= pct <= 1.0). Plain
    Python, no numpy — this log is realistically thousands of rows at
    most, and nearest-rank is a well-understood, if slightly less
    "smooth", choice compared to interpolated percentile methods; good
    enough for "is p95 latency creeping up" without pulling in a
    dependency for it.
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = min(int(len(sorted_values) * pct), len(sorted_values) - 1)
    return round(sorted_values[idx], 2)


def _summarize(records: List[Dict]) -> Dict:
    """Shared summary shape for both the overall totals and each by-mode breakdown."""
    n = len(records)
    if n == 0:
        return {
            "count": 0,
            "error_count": 0,
            "hit_rate": 0.0,
            "avg_total_latency_ms": 0.0,
            "p50_total_latency_ms": 0.0,
            "p95_total_latency_ms": 0.0,
            "avg_retrieval_latency_ms": 0.0,
            "avg_generation_latency_ms": 0.0,
            "avg_prompt_tokens": 0.0,
            "avg_completion_tokens": 0.0,
        }

    errors = [r for r in records if r.get("error")]
    successes = [r for r in records if not r.get("error")]
    total_latencies = [r["total_latency_ms"] for r in records if "total_latency_ms" in r]

    return {
        "count": n,
        "error_count": len(errors),
        # hit_rate is only meaningful over successful queries — an error
        # record has hit_count=0 by construction (nothing was found, that
        # IS the error), so folding it into the denominator would make
        # hit_rate partly just re-measure the error rate instead.
        "hit_rate": round(
            sum(1 for r in successes if r.get("hit_count", 0) > 0) / len(successes), 4
        ) if successes else 0.0,
        "avg_total_latency_ms": _avg(total_latencies),
        "p50_total_latency_ms": _percentile(total_latencies, 0.50),
        "p95_total_latency_ms": _percentile(total_latencies, 0.95),
        "avg_retrieval_latency_ms": _avg([r["retrieval_latency_ms"] for r in records if "retrieval_latency_ms" in r]),
        "avg_generation_latency_ms": _avg([r["generation_latency_ms"] for r in successes if "generation_latency_ms" in r]),
        "avg_prompt_tokens": _avg([r["prompt_tokens_estimate"] for r in successes if "prompt_tokens_estimate" in r]),
        "avg_completion_tokens": _avg([r["completion_tokens_estimate"] for r in successes if "completion_tokens_estimate" in r]),
    }


def aggregate_query_metrics(records: List[Dict]) -> Dict:
    """
    Aggregates a list of query records (as returned by load_query_records)
    into overall totals plus a per-retrieval-mode breakdown — "avg latency
    by retrieval mode, hit-rate over time", directly.

    Pure function, no I/O — takes records in, returns a plain dict out, so
    it's testable with hand-built fixtures and reusable from both the
    /metrics API endpoint and anything else that wants the same numbers
    (a future Streamlit dashboard, a CLI, etc.) without re-implementing
    the aggregation math.
    """
    overall = _summarize(records)

    by_mode: Dict[str, Dict] = {}
    modes = sorted({r.get("retrieval_mode", "unknown") for r in records})
    for mode in modes:
        mode_records = [r for r in records if r.get("retrieval_mode", "unknown") == mode]
        by_mode[mode] = _summarize(mode_records)

    return {"overall": overall, "by_mode": by_mode}