"""
Persists eval run results with a timestamp, so retrieval/answer-quality
metrics can be tracked across runs instead of eyeballed one run at a time.
Answers "did that chunk-size change / reranker swap help or hurt?" by
comparing the current run against the most recent prior run for the same
mode(s), rather than requiring the user to remember (or re-run) the old
config to find out.

Format: JSON Lines (one JSON object per line), append-only. Chosen over
SQLite for this project: zero new dependencies, trivially diffable/
greppable/git-friendly, and every consumer here (a Python list of dicts)
is well served without needing a query language for a log that's
realistically a few hundred rows at most. If this ever needs proper
filtering/aggregation across thousands of runs, that's the moment to
switch to sqlite3 (also stdlib, also no new dependency) — noted here
rather than silently declaring JSONL "good enough forever".

Every function here is pure — no Chroma/Ollama dependency, so this whole
module is unit-testable without any live service.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "eval_runs.jsonl"

# config.py values worth snapshotting per run — anything that plausibly
# changes eval outcomes: chunking, retrieval tuning, reranker choice,
# query rewriting/expansion, and which models were used. Deliberately NOT
# everything in config.py — file-extension allowlists, upload size caps,
# etc. don't affect what a fixed, already-indexed eval collection returns.
CONFIG_SNAPSHOT_KEYS = [
    "OLLAMA_MODEL",
    "OLLAMA_EMBEDDING_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "MAX_RETRIEVAL_DOCS",
    "RETRIEVAL_FETCH_K",
    "MMR_LAMBDA",
    "RRF_K",
    "RERANKER_MODEL",
    "ENABLE_QUERY_REWRITE",
    "QUERY_REWRITE_HISTORY_TURNS",
    "ENABLE_QUERY_EXPANSION",
    "QUERY_EXPANSION_COUNT",
]

# Which results sub-sections diff_run_results compares. Keep in sync with
# the keys run_eval.py's run_mode() puts in each mode's result dict
# ("retrieval" always; "answers"/"judge" only when those flags are on).
_METRIC_SECTIONS = ("retrieval", "answers", "judge")


def snapshot_config(config_module) -> Dict:
    """
    Reads CONFIG_SNAPSHOT_KEYS off the given config module (pass the real
    `config` module for an actual run; tests can pass any object/namespace
    with the same attribute names). A key missing from the module is
    recorded as None rather than raising — config.py can gain or rename
    settings over time, and an old log entry should stay loadable rather
    than the whole snapshot step breaking on a renamed constant.
    """
    return {key: getattr(config_module, key, None) for key in CONFIG_SNAPSHOT_KEYS}


def build_run_record(
    *,
    collection: str,
    project_dir: str,
    k: int,
    modes: List[str],
    score_answers: bool,
    judge: bool,
    results_by_mode: Dict[str, dict],
    config_module,
) -> Dict:
    """Assembles one log entry — see the module docstring for the overall shape/rationale."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collection": collection,
        "project_dir": project_dir,
        "k": k,
        "modes": list(modes),
        "score_answers": score_answers,
        "judge": judge,
        "config": snapshot_config(config_module),
        "results": results_by_mode,
    }


def record_run(log_path, record: Dict) -> None:
    """Appends one run record as a single JSON line. Creates the log file (and parent dirs) if needed."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_runs(log_path) -> List[Dict]:
    """
    Reads every run record from the log, oldest first. A missing file
    returns an empty list (a fresh project with no eval history yet is
    not an error). A malformed line is logged and skipped rather than
    aborting the whole load — one corrupted line (e.g. from an
    interrupted write) shouldn't make every other run's history unreadable.
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
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed eval log line %d in %s", line_no, log_path)

    return records


def most_recent_run_for_mode(records: List[Dict], mode: str, before_index: Optional[int] = None) -> Optional[Dict]:
    """
    Most recent record (by list order — call load_runs() first, which
    returns oldest-first) that includes results for `mode`, searched from
    the end of `records` up to (but not including) `before_index` if
    given. Used to look up "the last run before this one" when comparing,
    rather than matching the run currently being logged against itself.
    Returns None if no earlier record covers that mode at all (e.g. the
    very first time this mode has ever been evaluated).
    """
    search_space = records[:before_index] if before_index is not None else records
    for record in reversed(search_space):
        if mode in record.get("results", {}):
            return record
    return None


def diff_run_results(current: Dict, previous: Dict) -> Dict[str, Dict[str, float]]:
    """
    For each metrics section present in both current and previous (one
    mode's "results" dict each — see _METRIC_SECTIONS), computes
    {metric_name: current_value - previous_value} for every numeric
    metric found in both. A metric present in only one side, or a
    non-numeric field, is silently skipped rather than raising — not
    every metric field is guaranteed to be a number to diff (and the set
    of fields can grow over time as new metrics get added).
    """
    diffs: Dict[str, Dict[str, float]] = {}

    for section in _METRIC_SECTIONS:
        cur_section = current.get(section)
        prev_section = previous.get(section)
        if not cur_section or not prev_section:
            continue

        section_diff = {}
        for key, cur_value in cur_section.items():
            if key not in prev_section:
                continue
            prev_value = prev_section[key]
            if _is_diffable_number(cur_value) and _is_diffable_number(prev_value):
                section_diff[key] = round(cur_value - prev_value, 4)

        if section_diff:
            diffs[section] = section_diff

    return diffs


def _is_diffable_number(value) -> bool:
    # bool is technically an int subclass in Python — exclude it so a
    # True/False field never gets silently treated as 1/0 and diffed.
    return isinstance(value, (int, float)) and not isinstance(value, bool)