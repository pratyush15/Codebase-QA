from core.query_log import (
    record_query,
    load_query_records,
    aggregate_query_metrics,
    _avg,
    _percentile,
    _summarize,
)


# ---------- record_query / load_query_records (round trip) ----------

def test_record_and_load_round_trip(tmp_path):
    log_path = tmp_path / "queries.jsonl"

    record_query(
        log_path, collection="proj", retrieval_mode="hybrid_rerank", hit_count=3,
        retrieval_latency_ms=100.0, generation_latency_ms=500.0, total_latency_ms=600.0,
        prompt_tokens_estimate=200, completion_tokens_estimate=50,
    )

    records = load_query_records(log_path)

    assert len(records) == 1
    r = records[0]
    assert r["collection"] == "proj"
    assert r["retrieval_mode"] == "hybrid_rerank"
    assert r["hit_count"] == 3
    assert r["total_latency_ms"] == 600.0
    assert r["error"] is False
    assert "timestamp" in r


def test_record_query_appends_not_overwrites(tmp_path):
    log_path = tmp_path / "queries.jsonl"
    record_query(log_path, collection="a", retrieval_mode="vector")
    record_query(log_path, collection="b", retrieval_mode="vector")

    records = load_query_records(log_path)

    assert [r["collection"] for r in records] == ["a", "b"]


def test_record_query_creates_parent_directories(tmp_path):
    log_path = tmp_path / "nested" / "dir" / "queries.jsonl"
    record_query(log_path, collection="proj", retrieval_mode="vector")
    assert log_path.exists()


def test_record_query_never_raises_on_write_failure(tmp_path):
    # Point at a path that can't possibly be written to (a file used as a
    # directory component) — record_query must swallow this, not raise.
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    bad_path = blocker / "queries.jsonl"

    record_query(bad_path, collection="proj", retrieval_mode="vector")  # must not raise


def test_record_query_error_case_defaults():
    pass  # covered by test_record_error_query below


def test_record_error_query_sets_error_true_and_zero_hit_count(tmp_path):
    log_path = tmp_path / "queries.jsonl"
    record_query(log_path, collection="proj", retrieval_mode="hybrid", error=True)

    records = load_query_records(log_path)
    assert records[0]["error"] is True
    assert records[0]["hit_count"] == 0


def test_record_query_never_includes_question_or_answer_text(tmp_path):
    # Explicit guard for the module's stated privacy contract: record_query
    # doesn't even accept a question/answer parameter, so this is really
    # checking the on-disk shape has no such field under any key.
    log_path = tmp_path / "queries.jsonl"
    record_query(log_path, collection="proj", retrieval_mode="vector")

    raw = log_path.read_text()
    assert "question" not in raw
    assert "answer" not in raw


# ---------- load_query_records ----------

def test_load_query_records_missing_file_returns_empty_list(tmp_path):
    assert load_query_records(tmp_path / "does_not_exist.jsonl") == []


def test_load_query_records_skips_malformed_lines(tmp_path):
    log_path = tmp_path / "queries.jsonl"
    log_path.write_text('{"collection": "a"}\nnot valid json\n{"collection": "b"}\n')

    records = load_query_records(log_path)

    assert [r["collection"] for r in records] == ["a", "b"]


def test_load_query_records_skips_blank_lines(tmp_path):
    log_path = tmp_path / "queries.jsonl"
    log_path.write_text('{"collection": "a"}\n\n\n{"collection": "b"}\n')

    records = load_query_records(log_path)

    assert [r["collection"] for r in records] == ["a", "b"]


def test_load_query_records_since_filters_by_timestamp(tmp_path):
    from datetime import datetime, timezone

    log_path = tmp_path / "queries.jsonl"
    log_path.write_text(
        '{"timestamp": "2026-01-01T00:00:00+00:00", "collection": "old"}\n'
        '{"timestamp": "2026-06-01T00:00:00+00:00", "collection": "new"}\n'
    )

    since = datetime(2026, 3, 1, tzinfo=timezone.utc)
    records = load_query_records(log_path, since=since)

    assert [r["collection"] for r in records] == ["new"]


def test_load_query_records_since_none_returns_everything(tmp_path):
    log_path = tmp_path / "queries.jsonl"
    log_path.write_text(
        '{"timestamp": "2026-01-01T00:00:00+00:00", "collection": "old"}\n'
        '{"timestamp": "2026-06-01T00:00:00+00:00", "collection": "new"}\n'
    )

    records = load_query_records(log_path, since=None)

    assert len(records) == 2


def test_load_query_records_skips_records_with_bad_timestamp_when_since_given(tmp_path):
    from datetime import datetime, timezone

    log_path = tmp_path / "queries.jsonl"
    log_path.write_text(
        '{"collection": "no_timestamp_field"}\n'
        '{"timestamp": "not-a-real-timestamp", "collection": "bad_ts"}\n'
        '{"timestamp": "2026-06-01T00:00:00+00:00", "collection": "good"}\n'
    )

    since = datetime(2026, 1, 1, tzinfo=timezone.utc)
    records = load_query_records(log_path, since=since)

    assert [r["collection"] for r in records] == ["good"]


# ---------- _avg / _percentile (pure math helpers) ----------

def test_avg_empty_list_is_zero():
    assert _avg([]) == 0.0


def test_avg_computes_mean():
    assert _avg([1.0, 2.0, 3.0]) == 2.0


def test_percentile_empty_list_is_zero():
    assert _percentile([], 0.5) == 0.0


def test_percentile_p50_of_sorted_values():
    values = [10, 20, 30, 40, 50]
    assert _percentile(values, 0.5) == 30


def test_percentile_p95_is_near_the_top():
    values = list(range(1, 101))  # 1..100
    p95 = _percentile(values, 0.95)
    assert p95 >= 95


def test_percentile_single_value():
    assert _percentile([42.0], 0.95) == 42.0


def test_percentile_unsorted_input_still_correct():
    values = [50, 10, 30, 20, 40]
    assert _percentile(values, 0.0) == 10  # p0 = minimum


# ---------- _summarize ----------

def test_summarize_empty_records_returns_zeroed_shape():
    result = _summarize([])
    assert result["count"] == 0
    assert result["hit_rate"] == 0.0
    assert result["avg_total_latency_ms"] == 0.0


def test_summarize_computes_hit_rate_over_successes_only():
    records = [
        {"error": False, "hit_count": 3, "total_latency_ms": 100},
        {"error": False, "hit_count": 0, "total_latency_ms": 100},  # success but zero hits (edge case)
        {"error": True, "hit_count": 0, "total_latency_ms": 50},
    ]
    result = _summarize(records)

    # hit_rate is over the 2 successful (non-error) records: 1 of them has hit_count > 0
    assert result["count"] == 3
    assert result["error_count"] == 1
    assert result["hit_rate"] == 0.5


def test_summarize_all_errors_hit_rate_is_zero_not_divide_by_zero():
    records = [{"error": True, "hit_count": 0, "total_latency_ms": 50}]
    result = _summarize(records)
    assert result["hit_rate"] == 0.0
    assert result["error_count"] == 1


def test_summarize_generation_latency_excludes_error_records():
    # Error records never ran generation, so including a 0 for them would
    # artificially drag the average down.
    records = [
        {"error": False, "hit_count": 1, "total_latency_ms": 100, "generation_latency_ms": 200},
        {"error": True, "hit_count": 0, "total_latency_ms": 20, "retrieval_latency_ms": 20},
    ]
    result = _summarize(records)
    assert result["avg_generation_latency_ms"] == 200.0


def test_summarize_retrieval_latency_includes_error_records():
    # Unlike generation, retrieval DOES run even on the "no results" path
    # — an error record still has a real retrieval_latency_ms.
    records = [
        {"error": False, "hit_count": 1, "total_latency_ms": 100, "retrieval_latency_ms": 80},
        {"error": True, "hit_count": 0, "total_latency_ms": 20, "retrieval_latency_ms": 20},
    ]
    result = _summarize(records)
    assert result["avg_retrieval_latency_ms"] == 50.0


# ---------- aggregate_query_metrics ----------

def test_aggregate_query_metrics_empty_list():
    result = aggregate_query_metrics([])
    assert result["overall"]["count"] == 0
    assert result["by_mode"] == {}


def test_aggregate_query_metrics_breaks_down_by_mode():
    records = [
        {"error": False, "hit_count": 1, "retrieval_mode": "vector", "total_latency_ms": 100},
        {"error": False, "hit_count": 1, "retrieval_mode": "hybrid_rerank", "total_latency_ms": 300},
        {"error": False, "hit_count": 0, "retrieval_mode": "hybrid_rerank", "total_latency_ms": 300},
    ]
    result = aggregate_query_metrics(records)

    assert result["overall"]["count"] == 3
    assert set(result["by_mode"].keys()) == {"vector", "hybrid_rerank"}
    assert result["by_mode"]["vector"]["count"] == 1
    assert result["by_mode"]["hybrid_rerank"]["count"] == 2
    assert result["by_mode"]["hybrid_rerank"]["hit_rate"] == 0.5


def test_aggregate_query_metrics_missing_mode_field_grouped_as_unknown():
    records = [{"error": False, "hit_count": 1, "total_latency_ms": 100}]  # no "retrieval_mode" key
    result = aggregate_query_metrics(records)
    assert "unknown" in result["by_mode"]


def test_aggregate_query_metrics_overall_matches_full_record_set():
    records = [
        {"error": False, "hit_count": 1, "retrieval_mode": "vector", "total_latency_ms": 100},
        {"error": False, "hit_count": 1, "retrieval_mode": "hybrid", "total_latency_ms": 200},
    ]
    result = aggregate_query_metrics(records)
    assert result["overall"]["count"] == 2
    assert result["overall"]["avg_total_latency_ms"] == 150.0