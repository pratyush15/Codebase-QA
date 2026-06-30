from typing import Dict, Generator
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from core.retriever import retrieve_documents, assemble_context, get_source_summary


SYSTEM_PROMPT = """You are an expert code assistant helping developers understand a codebase.
You will be given relevant code snippets from the codebase and a question about them.

Guidelines:
- Answer based strictly on the provided code context
- If the answer is not in the context, say so clearly — do not hallucinate
- When referencing code, use proper markdown code blocks with the correct language identifier
- Be concise but thorough — explain the why, not just the what
- If asked about a function/class/module, explain its purpose, inputs, outputs, and any side effects
- If the question spans multiple files, explain how they interact
"""

QA_PROMPT = PromptTemplate(
    input_variables=["system", "context", "question"],
    template="""
{system}

## Codebase Context
{context}

## Question
{question}

## Answer
""",
)


def get_llm() -> Ollama:
    return Ollama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.1,
    )


def run_qa_streaming(
    collection_name: str,
    question: str,
    filter_language: str = None,
    filter_source: str = None,
) -> Generator:
    docs = retrieve_documents(
        collection_name=collection_name,
        query=question,
        filter_language=filter_language,
        filter_source=filter_source,
    )

    if not docs:
        yield {"type": "error", "message": "No relevant code found. Try rephrasing or select the correct project."}
        return

    context = assemble_context(docs)
    sources = get_source_summary(docs)

    prompt_text = QA_PROMPT.format(
        system=SYSTEM_PROMPT,
        context=context,
        question=question,
    )

    llm = get_llm()
    for chunk in llm.stream(prompt_text):
        yield {"type": "token", "value": chunk}

    yield {"type": "sources", "value": sources}


def check_ollama_connection() -> Dict:
    try:
        llm = get_llm()
        llm.invoke("ping")
        return {"status": "ok", "model": OLLAMA_MODEL}
    except Exception as e:
        return {"status": "error", "message": str(e), "model": OLLAMA_MODEL}