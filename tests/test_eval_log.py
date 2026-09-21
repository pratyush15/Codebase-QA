import sys
from pathlib import Path
from types import SimpleNamespace

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL_DIR))

from eval_log import (  # noqa: E402
    snapshot_config,
    build_run_record,
    record_run,
    load_runs,
    most_recent_run_for_mode,
    diff_run_results,
    CONFIG_SNAPSHOT_KEYS,
)


# ---------- snapshot_config ----------

def test_snapshot_config_reads_all_expected_keys():
    fake_config = SimpleNamespace(**{k: "x" for k in CONFIG_SNAPSHOT_KEYS})
    result = snapshot_config(fake_config)
    assert set(result.keys()) == set(CONFIG_SNAPSHOT_KEYS)
    assert all(v == "x" for v in result.values())


def test_snapshot_config_missing_attribute_becomes_none():
    fake_config = SimpleNamespace()  # nothing set
    result = snapshot_config(fake_config)
    assert all(v is None for v in result.values())


def test_snapshot_config_partial_module():
    fake_config = SimpleNamespace(CHUNK_SIZE=1000, RERANKER_MODEL="some-model")
    result = snapshot_config(fake_config)
    assert result["CHUNK_SIZE"] == 1000
    assert result["RERANKER_MODEL"] == "some-model"
    assert result["OLLAMA_MODEL"] is None


# ---------- build_run_record ----------

def test_build_run_record_shape():
    fake_config = SimpleNamespace(CHUNK_SIZE=1000)
    record = build_run_record(
        collection="eval-self",
        project_dir="/some/path",
        k=6,
        modes=["vector", "hybrid"],
        score_answers=True,
        judge=False,
        results_by_mode={"vector": {"retrieval": {"hit_rate": 0.9}}},
        config_module=fake_config,
    )

    assert record["collection"] == "eval-self"
    assert record["project_dir"] == "/some/path"
    assert record["k"] == 6
    assert record["modes"] == ["vector", "hybrid"]
    assert record["score_answers"] is True
    assert record["judge"] is False
    assert record["config"]["CHUNK_SIZE"] == 1000
    assert record["results"]["vector"]["retrieval"]["hit_rate"] == 0.9
    assert "timestamp" in record


def test_build_run_record_timestamp_is_iso_format_with_timezone():
    fake_config = SimpleNamespace()
    record = build_run_record(
        collection="c", project_dir="p", k=6, modes=[], score_answers=False,
        judge=False, results_by_mode={}, config_module=fake_config,
    )
    # Should be parseable and timezone-aware (ends with +00:00 for UTC).
    from datetime import datetime
    parsed = datetime.fromisoformat(record["timestamp"])
    assert parsed.tzinfo is not None


# ---------- record_run / load_runs (round trip) ----------

def test_record_and_load_round_trip(tmp_path):
    log_path = tmp_path / "runs.jsonl"
    record = {"timestamp": "2026-01-01T00:00:00+00:00", "modes": ["vector"], "results": {}}

    record_run(log_path, record)
    loaded = load_runs(log_path)

    assert loaded == [record]


def test_record_run_appends_not_overwrites(tmp_path):
    log_path = tmp_path / "runs.jsonl"
    record_run(log_path, {"id": 1})
    record_run(log_path, {"id": 2})

    loaded = load_runs(log_path)

    assert [r["id"] for r in loaded] == [1, 2]


def test_record_run_creates_parent_directories(tmp_path):
    log_path = tmp_path / "nested" / "dir" / "runs.jsonl"
    record_run(log_path, {"id": 1})
    assert log_path.exists()


def test_load_runs_missing_file_returns_empty_list(tmp_path):
    assert load_runs(tmp_path / "does_not_exist.jsonl") == []


def test_load_runs_skips_malformed_lines(tmp_path):
    log_path = tmp_path / "runs.jsonl"
    log_path.write_text('{"id": 1}\nnot valid json\n{"id": 2}\n')

    loaded = load_runs(log_path)

    assert [r["id"] for r in loaded] == [1, 2]


def test_load_runs_skips_blank_lines(tmp_path):
    log_path = tmp_path / "runs.jsonl"
    log_path.write_text('{"id": 1}\n\n\n{"id": 2}\n')

    loaded = load_runs(log_path)

    assert [r["id"] for r in loaded] == [1, 2]


def test_load_runs_preserves_order():
    pass  # covered by test_record_run_appends_not_overwrites


# ---------- most_recent_run_for_mode ----------

def test_most_recent_run_for_mode_finds_latest_matching():
    records = [
        {"timestamp": "t1", "results": {"vector": {}}},
        {"timestamp": "t2", "results": {"hybrid": {}}},
        {"timestamp": "t3", "results": {"vector": {}}},
    ]
    result = most_recent_run_for_mode(records, "vector")
    assert result["timestamp"] == "t3"


def test_most_recent_run_for_mode_returns_none_when_never_run():
    records = [{"timestamp": "t1", "results": {"vector": {}}}]
    assert most_recent_run_for_mode(records, "hybrid_rerank") is None


def test_most_recent_run_for_mode_returns_none_for_empty_history():
    assert most_recent_run_for_mode([], "vector") is None


def test_most_recent_run_for_mode_respects_before_index():
    records = [
        {"timestamp": "t1", "results": {"vector": {}}},
        {"timestamp": "t2", "results": {"vector": {}}},
        {"timestamp": "t3", "results": {"vector": {}}},  # the "current" run being logged
    ]
    # Searching before index 2 (the current run) should skip t3 and find t2.
    result = most_recent_run_for_mode(records, "vector", before_index=2)
    assert result["timestamp"] == "t2"


def test_most_recent_run_for_mode_missing_results_key_treated_as_no_match():
    records = [{"timestamp": "t1"}]  # no "results" key at all
    assert most_recent_run_for_mode(records, "vector") is None


# ---------- diff_run_results ----------

def test_diff_run_results_computes_deltas_for_shared_numeric_metrics():
    current = {"retrieval": {"hit_rate": 0.9, "mrr": 0.7}}
    previous = {"retrieval": {"hit_rate": 0.8, "mrr": 0.7}}

    diff = diff_run_results(current, previous)

    assert diff["retrieval"]["hit_rate"] == round(0.9 - 0.8, 4)
    assert diff["retrieval"]["mrr"] == 0.0


def test_diff_run_results_covers_multiple_sections():
    current = {
        "retrieval": {"hit_rate": 1.0},
        "answers": {"keyword_hit_rate": 0.8},
        "judge": {"avg_judge_score": 0.6},
    }
    previous = {
        "retrieval": {"hit_rate": 0.5},
        "answers": {"keyword_hit_rate": 0.6},
        "judge": {"avg_judge_score": 0.4},
    }

    diff = diff_run_results(current, previous)

    assert diff["retrieval"]["hit_rate"] == 0.5
    assert diff["answers"]["keyword_hit_rate"] == round(0.8 - 0.6, 4)
    assert diff["judge"]["avg_judge_score"] == round(0.6 - 0.4, 4)


def test_diff_run_results_skips_section_missing_from_either_side():
    current = {"retrieval": {"hit_rate": 1.0}, "answers": {"keyword_hit_rate": 0.8}}
    previous = {"retrieval": {"hit_rate": 0.5}}  # no "answers" section

    diff = diff_run_results(current, previous)

    assert "retrieval" in diff
    assert "answers" not in diff


def test_diff_run_results_skips_metric_missing_from_either_side():
    current = {"retrieval": {"hit_rate": 1.0, "new_metric": 5}}
    previous = {"retrieval": {"hit_rate": 0.5}}  # no "new_metric"

    diff = diff_run_results(current, previous)

    assert "hit_rate" in diff["retrieval"]
    assert "new_metric" not in diff["retrieval"]


def test_diff_run_results_skips_non_numeric_fields():
    current = {"judge": {"avg_judge_score": 0.9, "verdict_notes": "looks good"}}
    previous = {"judge": {"avg_judge_score": 0.5, "verdict_notes": "meh"}}

    diff = diff_run_results(current, previous)

    assert "avg_judge_score" in diff["judge"]
    assert "verdict_notes" not in diff["judge"]


def test_diff_run_results_excludes_booleans_even_though_they_are_ints():
    current = {"retrieval": {"hit_rate": 1.0, "some_flag": True}}
    previous = {"retrieval": {"hit_rate": 0.5, "some_flag": False}}

    diff = diff_run_results(current, previous)

    assert "hit_rate" in diff["retrieval"]
    assert "some_flag" not in diff["retrieval"]


def test_diff_run_results_empty_when_no_overlap_at_all():
    current = {"retrieval": {"hit_rate": 1.0}}
    previous = {}
    assert diff_run_results(current, previous) == {}


def test_diff_run_results_negative_delta_when_metric_got_worse():
    current = {"retrieval": {"hit_rate": 0.4}}
    previous = {"retrieval": {"hit_rate": 0.9}}

    diff = diff_run_results(current, previous)

    assert diff["retrieval"]["hit_rate"] == round(0.4 - 0.9, 4)
    assert diff["retrieval"]["hit_rate"] < 0