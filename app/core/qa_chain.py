
import time
import logging
from typing import Dict, Generator, List, Optional
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RETRIEVAL_MODE,
    MAX_RETRIEVAL_DOCS,
    ENABLE_QUERY_REWRITE,
    QUERY_REWRITE_HISTORY_TURNS,
    ENABLE_QUERY_EXPANSION,
    QUERY_EXPANSION_COUNT,
    ENABLE_QUERY_METRICS_LOG,
    QUERY_METRICS_LOG_PATH,
)
from core.retriever import retrieve_documents_advanced, assemble_context, get_source_summary
from core.query_rewrite import rewrite_query_standalone, format_recent_turns, expand_query
from core.query_log import record_query

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert code assistant helping developers understand a codebase.
You will be given relevant code snippets from the codebase and a question about them.

Guidelines:
- Answer based strictly on the provided code context
- If the answer is not in the context, say so clearly — do not hallucinate
- When referencing code, use proper markdown code blocks with the correct language identifier
- Be concise but thorough — explain the why, not just the what
- If asked about a function/class/module, explain its purpose, inputs, outputs, and any side effects
- If the question spans multiple files, explain how they interact
- If "Recent conversation" is provided below, use it only to understand what the question is
  referring to (e.g. "that function", "the other one") — the answer itself must still come
  strictly from the Codebase Context, not from anything said earlier in the conversation
"""

QA_PROMPT = PromptTemplate(
    input_variables=["system", "context", "conversation", "question"],
    template="""
{system}

## Codebase Context
{context}
{conversation}
## Question
{question}

## Answer
""",
)


def get_llm() -> OllamaLLM:
    return OllamaLLM(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.1,
    )


def _estimate_tokens(text: str) -> int:
    """
    Rough token estimate (~4 chars/token, a common heuristic for English/code
    text) since Ollama's streaming API here doesn't return exact token
    counts. Good enough for relative tracking (is this query unusually
    expensive?), not meant to be billing-accurate.
    """
    return max(1, round(len(text) / 4))


def run_qa_streaming(
    collection_name: str,
    question: str,
    filter_language: str = None,
    filter_source: str = None,
    filter_symbol: str = None,
    retrieval_mode: str = RETRIEVAL_MODE,
    chat_history: Optional[List[Dict]] = None,
    k: int = MAX_RETRIEVAL_DOCS,
) -> Generator:
    """
    k: number of chunks to retrieve. Defaults to config.MAX_RETRIEVAL_DOCS
    — the same value every real caller (Streamlit UI, /query API) already
    got implicitly before this parameter existed. Exposed mainly for
    eval/run_eval.py's --score-answers, so a --k override on the CLI
    applies consistently to both the retrieval-only metrics AND the
    full-pipeline answer-quality metrics in the same run, rather than the
    latter silently ignoring it in favor of this default.

    filter_symbol: exact match against a chunk's symbol_name metadata
    (see core.ast_chunking / core.retriever._build_filter), for scoping
    to a specific function/class/method — "show me the
    `retrieve_documents` function" — the same way filter_source scopes to
    a specific file.

    chat_history: prior turns in the current conversation, oldest first,
    each a dict with at least {"role": "user"|"assistant", "content": str}.
    Do NOT include the current `question` in it. Pass None/[] for a fresh
    conversation (e.g. the first question) — this skips the rewrite step
    entirely rather than paying an extra LLM call for nothing.
    """
    t_start = time.perf_counter()

    llm = get_llm()

    # Follow-up questions ("what about the class version?") share little
    # vocabulary with the code they're actually asking about, so retrieval
    # gets a query rewritten to stand on its own; the *answer* prompt still
    # uses the user's original wording plus a short conversation excerpt,
    # so the response reads naturally rather than answering the rewritten
    # question verbatim.
    search_query = question
    query_rewritten = False
    rewrite_latency_ms = 0.0
    if ENABLE_QUERY_REWRITE and chat_history:
        t_rewrite_start = time.perf_counter()
        search_query = rewrite_query_standalone(
            question, chat_history, llm, max_turns=QUERY_REWRITE_HISTORY_TURNS,
        )
        rewrite_latency_ms = round((time.perf_counter() - t_rewrite_start) * 1000, 1)
        query_rewritten = search_query != question
        if query_rewritten:
            logger.info("Query rewritten for retrieval: %r -> %r", question, search_query)

    # Multi-query expansion (RAG-Fusion): even a standalone question rarely
    # uses the same words as the code that answers it, so a few LLM
    # paraphrases of search_query get fused into hybrid retrieval alongside
    # it. Only applies to the hybrid/hybrid_rerank modes, since "vector"/
    # "mmr" aren't RRF-fusion-based to begin with.
    extra_queries: List[str] = []
    expansion_latency_ms = 0.0
    if ENABLE_QUERY_EXPANSION and retrieval_mode in ("hybrid", "hybrid_rerank"):
        t_expand_start = time.perf_counter()
        extra_queries = expand_query(search_query, llm, n=QUERY_EXPANSION_COUNT)
        expansion_latency_ms = round((time.perf_counter() - t_expand_start) * 1000, 1)
        if extra_queries:
            logger.info("Query expanded into %d paraphrase(s): %r", len(extra_queries), extra_queries)

    docs = retrieve_documents_advanced(
        collection_name=collection_name,
        query=search_query,
        k=k,
        mode=retrieval_mode,
        filter_language=filter_language,
        filter_source=filter_source,
        filter_symbol=filter_symbol,
        extra_queries=extra_queries or None,
    )

    t_retrieved = time.perf_counter()
    retrieval_latency_ms = round((t_retrieved - t_start) * 1000, 1)
    hit_count = len(docs)

    if not docs:
        logger.info(
            "qa_query collection=%s mode=%s hits=0 retrieval_ms=%.1f rewritten=%s expanded=%d — no results",
            collection_name, retrieval_mode, retrieval_latency_ms, query_rewritten, len(extra_queries),
        )
        if ENABLE_QUERY_METRICS_LOG:
            record_query(
                QUERY_METRICS_LOG_PATH,
                collection=collection_name,
                retrieval_mode=retrieval_mode,
                hit_count=0,
                retrieval_latency_ms=retrieval_latency_ms,
                query_rewritten=query_rewritten,
                query_expansion_count=len(extra_queries),
                error=True,
            )
        yield {"type": "error", "message": "No relevant code found. Try rephrasing or select the correct project."}
        return

    context = assemble_context(docs)
    sources = get_source_summary(docs)

    conversation_text = format_recent_turns(chat_history, max_turns=QUERY_REWRITE_HISTORY_TURNS)
    conversation_block = (
        f"\n## Recent conversation (context only — do not answer from this)\n{conversation_text}\n"
        if conversation_text else ""
    )

    prompt_text = QA_PROMPT.format(
        system=SYSTEM_PROMPT,
        context=context,
        conversation=conversation_block,
        question=question,
    )

    answer_text = ""
    for chunk in llm.stream(prompt_text):
        answer_text += chunk
        yield {"type": "token", "value": chunk}

    t_end = time.perf_counter()
    generation_latency_ms = round((t_end - t_retrieved) * 1000, 1)
    total_latency_ms = round((t_end - t_start) * 1000, 1)
    prompt_tokens_est = _estimate_tokens(prompt_text)
    completion_tokens_est = _estimate_tokens(answer_text)

    logger.info(
        "qa_query collection=%s mode=%s hits=%d rewritten=%s expanded=%d retrieval_ms=%.1f generation_ms=%.1f "
        "total_ms=%.1f prompt_tokens~=%d completion_tokens~=%d",
        collection_name, retrieval_mode, hit_count, query_rewritten, len(extra_queries), retrieval_latency_ms,
        generation_latency_ms, total_latency_ms, prompt_tokens_est, completion_tokens_est,
    )

    if ENABLE_QUERY_METRICS_LOG:
        record_query(
            QUERY_METRICS_LOG_PATH,
            collection=collection_name,
            retrieval_mode=retrieval_mode,
            hit_count=hit_count,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
            prompt_tokens_estimate=prompt_tokens_est,
            completion_tokens_estimate=completion_tokens_est,
            query_rewritten=query_rewritten,
            query_expansion_count=len(extra_queries),
            error=False,
        )

    yield {"type": "sources", "value": sources}
    yield {
        "type": "metrics",
        "value": {
            "retrieval_mode": retrieval_mode,
            "hit_count": hit_count,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "total_latency_ms": total_latency_ms,
            "prompt_tokens_estimate": prompt_tokens_est,
            "completion_tokens_estimate": completion_tokens_est,
            "query_rewritten": query_rewritten,
            "rewritten_query": search_query if query_rewritten else None,
            "query_rewrite_latency_ms": rewrite_latency_ms,
            "expanded_queries": extra_queries or None,
            "query_expansion_latency_ms": expansion_latency_ms,
        },
    }


def check_ollama_connection() -> Dict:
    try:
        llm = get_llm()
        llm.invoke("ping")
        return {"status": "ok", "model": OLLAMA_MODEL}
    except Exception as e:
        return {"status": "error", "message": str(e), "model": OLLAMA_MODEL}