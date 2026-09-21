
"""
Evaluation harness CLI.

Indexes this project's own codebase into a dedicated eval collection, runs
every question in eval/eval_set.py against one or more retrieval modes, and
reports precision@k / recall@k / MRR / hit-rate for each — so retrieval
quality changes can be measured, not just eyeballed.

That's the *retrieval* half: "did we find the right file?" It says
nothing about whether the LLM then actually answered correctly using that
file. --score-answers closes that loop by running the full retrieval ->
generation pipeline (same code path a real user hits) and scoring the
generated answer against eval_set.py's expected_keywords for that
question — see eval/answer_metrics.py. --judge additionally has a second
LLM call grade each answer correct/partial/incorrect — see
eval/llm_judge.py — catching a wrong or hallucinated explanation that
happens to still namedrop the right keywords, which keyword matching alone
can't. --judge implies --score-answers (there's no answer to judge
without generating one first).

Answer scoring runs at the same --k as the retrieval-only pass (defaulting
to config.MAX_RETRIEVAL_DOCS — the app's real out-of-the-box retrieval
depth — when --k isn't overridden), so an explicit --k applies
consistently to both halves of the report in a single run rather than the
retrieval table and the answer-quality table silently describing two
different retrieval depths.

Every run is logged (see eval/eval_log.py) to a JSON Lines file with a
timestamp and a snapshot of the retrieval/chunking config in effect — so
"did that chunk-size change / reranker swap help or hurt?" is a question
this tool can actually answer, rather than requiring you to remember or
re-run the old config yourself.
Each run's report includes a diff against the most recent prior run for
each mode. Use --history to just print past runs without running a new
eval, and --no-log to skip logging a particular run (e.g. a throwaway
manual check you don't want cluttering the history).

Requires a running Ollama instance (for embeddings, and for generation when
--score-answers/--judge are used) — same requirement as indexing/querying
through the normal app. This is the one piece of Tier 3 that can't be
exercised in a sandboxed CI environment without network access to a local
LLM; tests/test_eval_metrics.py, tests/test_answer_metrics.py,
tests/test_llm_judge.py, and tests/test_eval_log.py cover the scoring/
logging math itself with no live dependencies.

Usage (from the project root):
    python eval/run_eval.py
    python eval/run_eval.py --modes vector mmr hybrid hybrid_rerank
    python eval/run_eval.py --k 3
    python eval/run_eval.py --score-answers
    python eval/run_eval.py --score-answers --judge
    python eval/run_eval.py --history          # print the last 10 logged runs and exit
    python eval/run_eval.py --history 30       # print the last 30
    python eval/run_eval.py --no-log           # run without appending to the log
    python eval/run_eval.py --project-dir /path/to/other/codebase --collection other-eval
"""
import sys
import argparse
from pathlib import Path

# Make `app/` importable the same way api.py and main.py expect.
APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import config  # noqa: E402
from utils.file_handler import extract_files_from_directory  # noqa: E402
from core.indexing import sync_files_to_collection  # noqa: E402
from core.retriever import retrieve_documents_advanced  # noqa: E402
from core.vectorstore import delete_collection, collection_exists  # noqa: E402
from core.qa_chain import run_qa_streaming, get_llm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_set import EVAL_SET  # noqa: E402
from metrics import evaluate_single, aggregate_results  # noqa: E402
from answer_metrics import evaluate_answer, aggregate_answer_results  # noqa: E402
from llm_judge import judge_answer, aggregate_judge_results  # noqa: E402
from eval_log import (  # noqa: E402
    DEFAULT_LOG_PATH,
    build_run_record,
    record_run,
    load_runs,
    most_recent_run_for_mode,
    diff_run_results,
)

ALL_MODES = ["vector", "mmr", "hybrid", "hybrid_rerank"]


def index_project(project_dir: str, collection_name: str) -> None:
    print(f"Indexing '{project_dir}' into collection '{collection_name}'...")
    files = extract_files_from_directory(project_dir)
    if not files:
        sys.exit(f"No readable files found under {project_dir} — nothing to evaluate.")
    stats = sync_files_to_collection(collection_name, files)
    print(
        f"  {stats['new_files']} new, {stats['updated_files']} updated, "
        f"{stats['skipped_files']} unchanged ({stats['chunks_embedded']} chunks embedded)\n"
    )


def _generate_answer(collection_name: str, question: str, mode: str, k: int) -> tuple:
    """
    Runs the real answer-generation pipeline for one question and returns
    (answer_text, retrieved_sources) — the same event stream the Streamlit
    UI and /query API consume, just collected into a single result instead
    of streamed. An in-band "error" event (e.g. "no relevant code found")
    produces an empty answer, which then correctly scores as a miss on
    every answer-quality metric rather than raising and aborting the run.
    """
    answer_text = ""
    retrieved_sources: list = []

    for event in run_qa_streaming(collection_name=collection_name, question=question, retrieval_mode=mode, k=k):
        if event["type"] == "token":
            answer_text += event["value"]
        elif event["type"] == "sources":
            retrieved_sources = [s["source"] for s in event["value"]]
        elif event["type"] == "error":
            answer_text = ""
            retrieved_sources = []

    return answer_text, retrieved_sources


def run_mode(
    collection_name: str,
    mode: str,
    k: int,
    score_answers: bool = False,
    judge: bool = False,
    judge_llm=None,
) -> dict:
    per_question_retrieval = []
    per_question_answers = []
    per_question_judged = []

    for item in EVAL_SET:
        if score_answers:
            answer_text, retrieved_sources = _generate_answer(collection_name, item["question"], mode, k)
        else:
            docs = retrieve_documents_advanced(
                collection_name=collection_name, query=item["question"], k=k, mode=mode,
            )
            retrieved_sources = []
            for doc in docs:
                src = doc.metadata.get("source", "")
                if src not in retrieved_sources:
                    retrieved_sources.append(src)

        per_question_retrieval.append(evaluate_single(retrieved_sources, item["expected_sources"], k))

        if score_answers:
            per_question_answers.append(evaluate_answer(answer_text, item.get("expected_keywords", [])))
            if judge:
                per_question_judged.append(judge_answer(item["question"], answer_text, judge_llm))

    result = {"retrieval": aggregate_results(per_question_retrieval)}
    if score_answers:
        result["answers"] = aggregate_answer_results(per_question_answers)
    if judge:
        result["judge"] = aggregate_judge_results(per_question_judged)
    return result


def print_report(results_by_mode: dict, k: int, score_answers: bool, judge: bool):
    print(f"Retrieval eval — {len(EVAL_SET)} questions, k={k}\n")
    header = f"{'mode':<16}{'precision@k':<14}{'recall@k':<12}{'MRR':<10}{'hit rate':<10}"
    print(header)
    print("-" * len(header))
    for mode, r in results_by_mode.items():
        ret = r["retrieval"]
        print(f"{mode:<16}{ret['precision_at_k']:<14}{ret['recall_at_k']:<12}{ret['mrr']:<10}{ret['hit_rate']:<10}")

    if score_answers:
        print(f"\nAnswer-quality eval (keyword check) — {len(EVAL_SET)} questions\n")
        header = f"{'mode':<16}{'keyword coverage':<20}{'keyword hit rate':<18}"
        print(header)
        print("-" * len(header))
        for mode, r in results_by_mode.items():
            a = r["answers"]
            print(f"{mode:<16}{a['keyword_coverage']:<20}{a['keyword_hit_rate']:<18}")

    if judge:
        print(f"\nAnswer-quality eval (LLM judge) — {len(EVAL_SET)} questions\n")
        header = f"{'mode':<16}{'avg score':<12}{'correct':<10}{'partial':<10}{'incorrect':<11}{'error':<8}"
        print(header)
        print("-" * len(header))
        for mode, r in results_by_mode.items():
            j = r["judge"]
            print(
                f"{mode:<16}{j['avg_judge_score']:<12}{j['correct']:<10}{j['partial']:<10}"
                f"{j['incorrect']:<11}{j['error']:<8}"
            )


def print_diff_report(results_by_mode: dict, previous_records: list) -> None:
    """
    For each mode just run, looks up the most recent prior logged run that
    also covered that mode and prints the metric deltas — this is the
    actual "did that config change help or hurt" answer, rather than
    requiring the reader to mentally diff two separate table printouts
    (or worse, two separate terminal scrollback sessions from different days).
    """
    any_diff_printed = False

    for mode, current_result in results_by_mode.items():
        previous = most_recent_run_for_mode(previous_records, mode)
        if previous is None:
            continue

        previous_result = previous.get("results", {}).get(mode, {})
        diff = diff_run_results(current_result, previous_result)
        if not diff:
            continue

        if not any_diff_printed:
            print("\nvs. previous logged run:\n")
            any_diff_printed = True

        prev_ts = (previous.get("timestamp") or "?")[:19]
        print(f"  {mode}  (vs {prev_ts}):")
        for section, metrics in diff.items():
            for metric, delta in metrics.items():
                sign = "+" if delta >= 0 else ""
                print(f"    {section}.{metric}: {sign}{delta}")

    if not any_diff_printed:
        print("\n(no prior logged run to compare against for these mode(s) — this is the first data point)")


def print_history(records: list, last_n: int) -> None:
    if not records:
        print("No eval runs logged yet — run without --history to create the first entry.")
        return

    recent = records[-last_n:]
    print(f"Last {len(recent)} of {len(records)} logged eval run(s):\n")
    header = f"{'timestamp':<21}{'mode':<16}{'hit rate':<10}{'MRR':<8}{'kw hit':<8}{'judge':<8}"
    print(header)
    print("-" * len(header))

    for record in recent:
        ts = (record.get("timestamp") or "")[:19]
        for mode, r in record.get("results", {}).items():
            ret = r.get("retrieval", {})
            ans = r.get("answers", {})
            j = r.get("judge", {})
            hit_rate = ret.get("hit_rate", "-")
            mrr = ret.get("mrr", "-")
            kw = ans.get("keyword_hit_rate", "-")
            js = j.get("avg_judge_score", "-")
            print(f"{ts:<21}{mode:<16}{hit_rate!s:<10}{mrr!s:<8}{kw!s:<8}{js!s:<8}")


def main():
    parser = argparse.ArgumentParser(description="Retrieval + answer-quality evaluation harness")
    parser.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--project-dir", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--collection", default="eval-codebase-qa-self")
    parser.add_argument("--reindex", action="store_true", help="Delete and rebuild the eval collection before running.")
    parser.add_argument(
        "--score-answers", action="store_true",
        help="Also run full answer generation and score it against eval_set.py's expected_keywords "
             "(keyword-match check, no extra LLM call beyond generation itself).",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Also grade each generated answer correct/partial/incorrect with a second LLM call "
             "(implies --score-answers). Roughly doubles per-question eval time.",
    )
    parser.add_argument(
        "--log-path", default=str(DEFAULT_LOG_PATH),
        help=f"JSON Lines file to append run results to (default: {DEFAULT_LOG_PATH}).",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Don't append this run's results to the eval log — for a throwaway manual check.",
    )
    parser.add_argument(
        "--history", type=int, nargs="?", const=10, default=None, metavar="N",
        help="Print the last N logged runs (default 10) and exit, without running a new eval.",
    )
    args = parser.parse_args()

    if args.history is not None:
        print_history(load_runs(args.log_path), args.history)
        return

    score_answers = args.score_answers or args.judge

    if args.reindex and collection_exists(args.collection):
        delete_collection(args.collection)

    index_project(args.project_dir, args.collection)

    judge_llm = get_llm() if args.judge else None

    results_by_mode = {}
    for mode in args.modes:
        print(f"Running mode: {mode}...")
        results_by_mode[mode] = run_mode(
            args.collection, mode, args.k,
            score_answers=score_answers, judge=args.judge, judge_llm=judge_llm,
        )

    print()
    print_report(results_by_mode, args.k, score_answers=score_answers, judge=args.judge)

    if not args.no_log:
        # Load history BEFORE appending this run, so the diff compares
        # against what came before it, not against itself.
        previous_records = load_runs(args.log_path)
        print_diff_report(results_by_mode, previous_records)

        record = build_run_record(
            collection=args.collection,
            project_dir=args.project_dir,
            k=args.k,
            modes=args.modes,
            score_answers=score_answers,
            judge=args.judge,
            results_by_mode=results_by_mode,
            config_module=config,
        )
        record_run(args.log_path, record)
        print(f"\nLogged this run to {args.log_path}")


if __name__ == "__main__":
    main()