from unittest.mock import patch

from langchain_core.documents import Document

from core.reranker import rerank


def _doc(source, content):
    return Document(page_content=content, metadata={"source": source})


def test_rerank_reorders_by_mocked_cross_encoder_score():
    docs = [_doc("low.py", "irrelevant"), _doc("high.py", "very relevant match")]

    class FakeModel:
        def predict(self, pairs):
            # Score the second pair (high.py) higher than the first.
            return [0.1, 0.9]

    with patch("core.reranker._get_model", return_value=FakeModel()), \
         patch("core.reranker.SENTENCE_TRANSFORMERS_AVAILABLE", True):
        result = rerank("query", docs, top_k=2)

    assert [d.metadata["source"] for d in result] == ["high.py", "low.py"]


def test_rerank_respects_top_k():
    docs = [_doc(f"f{i}.py", "x") for i in range(5)]

    class FakeModel:
        def predict(self, pairs):
            return list(range(len(pairs)))

    with patch("core.reranker._get_model", return_value=FakeModel()), \
         patch("core.reranker.SENTENCE_TRANSFORMERS_AVAILABLE", True):
        result = rerank("query", docs, top_k=2)

    assert len(result) == 2


def test_rerank_returns_empty_for_no_docs():
    assert rerank("query", [], top_k=5) == []


def test_rerank_falls_back_to_unranked_when_library_unavailable():
    docs = [_doc(f"f{i}.py", "x") for i in range(3)]
    with patch("core.reranker.SENTENCE_TRANSFORMERS_AVAILABLE", False):
        result = rerank("query", docs, top_k=2)
    assert result == docs[:2]


def test_rerank_falls_back_to_unranked_on_model_failure():
    docs = [_doc(f"f{i}.py", "x") for i in range(3)]

    def _boom(model_name):
        raise RuntimeError("no network")

    with patch("core.reranker._get_model", side_effect=_boom), \
         patch("core.reranker.SENTENCE_TRANSFORMERS_AVAILABLE", True):
        result = rerank("query", docs, top_k=2)

    assert result == docs[:2]  # graceful fallback, not a crash