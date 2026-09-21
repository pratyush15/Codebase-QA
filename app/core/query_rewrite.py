
"""
LLM-based query transformation, ahead of retrieval.

Two independent techniques live here:

1. rewrite_query_standalone() — conversation-aware rewriting. A follow-up
   like "what about the class version?" shares almost no vocabulary with
   the code it's actually asking about; this resolves references against
   recent chat history into a standalone question. 

2. expand_query() — multi-query expansion (a.k.a. RAG-Fusion). Even a
   standalone, well-formed question often uses different words than the
   code that answers it ("log a user in" vs. a codebase that says
   `authenticate_session`). This asks the LLM for a few alternative
   phrasings of the question, which core.hybrid_retriever.hybrid_retrieve_multi_query
   then runs retrieval against *in addition to* the original — a chunk
   that only matches one phrasing's wording still gets a fair shot, and
   fused rankings let phrasings the retriever's tokenizer/embedding
   happens to like carry the query rather than betting everything on how
   the user originally typed it.
"""
import re
import logging
from typing import Dict, List, Optional

from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


REWRITE_PROMPT = PromptTemplate(
    input_variables=["history", "question"],
    template="""Given the conversation history below, rewrite the "Latest question" as a standalone question that can be understood with no other context — resolve any pronouns or references (e.g. "that", "it", "the class version", "why does it do that") using the history.

Rules:
- If the latest question is already standalone, output it unchanged.
- Preserve the original intent and phrasing as closely as possible. Do not answer the question — only rewrite it.
- Output ONLY the rewritten question. No preamble, no quotes, no explanation.

## Conversation history
{history}

## Latest question
{question}

## Standalone question
""",
)


EXPANSION_PROMPT = PromptTemplate(
    input_variables=["question", "n"],
    template="""Generate exactly {n} alternative phrasings of the "Original question" below, to be used as additional search queries against a codebase.

Rules:
- Preserve the original intent, but use different wording — synonyms, more/less technical terms, or terminology likely to match actual code identifiers (e.g. "user login" -> "authenticate user session", "check who's signed in").
- Keep each phrasing short: a search query, not a full sentence or an answer.
- Each phrasing must be meaningfully different from the others, not trivial rewordings of each other.
- Output ONLY the {n} phrasings, one per line, numbered "1.", "2.", etc. Nothing else — no preamble, no explanation.

## Original question
{question}

## Alternative phrasings
""",
)


def format_recent_turns(history: Optional[List[Dict]], max_turns: int = 3) -> str:
    """
    Renders the last `max_turns` (user, assistant) turn-pairs as plain
    "Role: content" lines, for either the rewrite prompt or as light
    conversational context in the answer prompt. Long assistant answers
    are truncated — this is for resolving references, not re-feeding
    entire prior answers back into the model.
    """
    if not history:
        return ""

    trimmed = history[-max_turns * 2:] if max_turns else history
    lines = []
    for turn in trimmed:
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        role = "User" if turn.get("role") == "user" else "Assistant"
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def rewrite_query_standalone(
    question: str,
    history: Optional[List[Dict]],
    llm,
    max_turns: int = 3,
) -> str:
    """
    Returns a standalone version of `question` for retrieval purposes.

    Args:
        question: the latest user question, as typed.
        history: prior turns, oldest first, each a dict with at least
            "role" ("user"/"assistant") and "content". Pass the messages
            that preceded this question — not including it.
        llm: any object exposing .invoke(prompt_text) -> str (e.g. the
            same OllamaLLM instance qa_chain uses for answering).
        max_turns: how many trailing (user, assistant) pairs to consider.

    Falls back to the original `question`, unmodified, whenever there's no
    usable history or the rewrite call raises — never lets a rewrite
    failure surface as a query error.
    """
    history_text = format_recent_turns(history, max_turns)
    if not history_text:
        return question

    prompt_text = REWRITE_PROMPT.format(history=history_text, question=question)

    try:
        rewritten = llm.invoke(prompt_text)
    except Exception:
        logger.exception("Query rewrite failed — falling back to original question")
        return question

    rewritten = (rewritten or "").strip().strip('"').strip()
    if not rewritten:
        return question

    return rewritten


_LIST_ITEM_RE = re.compile(r"^\s*(?:\d+[\.\)]|[-*•])\s*(.+)$")


def _parse_numbered_list(text: str, expected_count: int) -> List[str]:
    """
    Pulls up to `expected_count` items out of LLM output shaped like a
    numbered/bulleted list ("1. foo", "2) bar", "- baz"). Falls back to
    treating each non-empty line as its own item if the model didn't
    number them — LLM output on "output only a list" instructions is
    reliable in spirit, not always in exact formatting.
    """
    if not text:
        return []

    items: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _LIST_ITEM_RE.match(line)
        content = match.group(1) if match else line
        content = content.strip().strip('"').strip()
        if content:
            items.append(content)
        if len(items) >= expected_count:
            break

    return items


def expand_query(question: str, llm, n: int = 2) -> List[str]:
    """
    Returns up to `n` alternative phrasings of `question`, generated by
    the LLM — for fusing into hybrid retrieval alongside the original
    (see core.hybrid_retriever.hybrid_retrieve_multi_query). Does NOT
    include `question` itself in the returned list.

    Returns an empty list (never raises) if n <= 0, the LLM call fails, or
    the model's output doesn't yield any usable paraphrases — callers
    should treat an empty list as "just use the original query", exactly
    like retrieval behaves without expansion at all.
    """
    if n <= 0:
        return []

    prompt_text = EXPANSION_PROMPT.format(question=question, n=n)

    try:
        raw = llm.invoke(prompt_text)
    except Exception:
        logger.exception("Query expansion failed — continuing with the original query only")
        return []

    candidates = _parse_numbered_list(raw or "", n)

    seen = {question.strip().lower()}
    paraphrases: List[str] = []
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        paraphrases.append(candidate)
        if len(paraphrases) >= n:
            break

    return paraphrases