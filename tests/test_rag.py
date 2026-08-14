"""Smoke tests for the RAG mode — no network / LLM calls required."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np
import pytest


def test_rag_config():
    from membukkit.config import RAGConfig

    cfg = RAGConfig()
    assert cfg.method == "coremem"
    assert cfg.fusion == "cosine"
    assert cfg.top_k == 5
    assert cfg.decompose is False


def test_rag_imports():
    from membukkit.rag import RAGSystem, RAGResult, RAGTrace

    assert RAGSystem is not None
    assert RAGResult is not None


def test_qa_scorer():
    from membukkit.eval.qa_scorer import score_qa, exact_match, f1

    assert exact_match(["Paris"], "Paris") == 1.0
    assert exact_match(["Paris"], "London") == 0.0
    assert f1(["the capital of France"], "France capital") > 0.0
    result = score_qa([["Paris"], ["London"]], ["Paris", "Berlin"])
    assert result["em"] == 50.0
    assert result["n"] == 2


def test_qa_scorer_recall():
    from membukkit.eval.qa_scorer import score_retrieval

    retrieved = [["A", "B", "C", "D", "E"], ["X", "Y", "Z", "W", "V"]]
    gold = [["A", "C"], ["Z", "Q"]]
    r = score_retrieval(retrieved, gold, ks=(2, 5))
    assert "recall@2" in r
    assert "recall@5" in r
    assert r["recall@5"] > 0


def test_dense_retriever_tiny():
    """Dense retriever on a tiny synthetic corpus (no real model, mocked encoder)."""
    from membukkit.retrieval.multihop import DenseRetriever

    class FakeEncoder:
        def encode_corpus(self, texts, batch_size=None):
            rng = np.random.default_rng(42)
            return rng.standard_normal((len(texts), 16)).astype(np.float32)

        def encode_query(self, query):
            rng = np.random.default_rng(99)
            return rng.standard_normal(16).astype(np.float32)

    enc = FakeEncoder()
    ret = DenseRetriever(enc)  # ty: ignore[invalid-argument-type]  # structural test double
    texts = [f"passage {i}" for i in range(10)]
    titles = [f"title {i}" for i in range(10)]
    ret.index(texts, titles)
    idxs = ret.retrieve("test query", top_k=3)
    assert len(idxs) == 3
    assert all(0 <= i < 10 for i in idxs)


def test_multihop_data_aliases():
    from membukkit.data.multihop import _ALIASES

    assert _ALIASES["musique"] == "musique"
    assert _ALIASES["2wiki"] == "2wikimultihopqa"
    assert _ALIASES["hotpot"] == "hotpotqa"


def test_qa_reader_parse():
    from membukkit.reading.qa_reader import _parse_answer

    assert _parse_answer("Thought: blah\nAnswer: Paris") == "Paris"
    assert _parse_answer("Paris") == "Paris"
    assert _parse_answer("") == ""


def test_interleave():
    from membukkit.retrieval.multihop import CoreMemRetriever

    lists = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    result = CoreMemRetriever._interleave(lists, top_k=6)
    assert len(result) == 6
    assert len(set(result)) == 6
    assert result[0] == 0
    assert result[1] == 3
    assert result[2] == 6


def test_bucketed_decomposition_routes_every_resolved_query_through_selector():
    """The unified path must not silently fall back to full-matrix cosine."""
    from membukkit.retrieval.multihop import CoreMemRetriever

    def fake_llm(prompt):
        if prompt.startswith("Decompose"):
            return "Who directed Example Film?\nWhat awards did the director win?"
        if "Who directed Example Film?" in prompt:
            return "Jane Doe"
        return "UNKNOWN"

    ret = CoreMemRetriever(
        encoder_path="unused",
        reranker_path="unused",
        decompose=True,
        decompose_retrieval="bucket",
        rerank_cap=4,
        llm_fn=fake_llm,
    )
    ret._enc = object()
    ret._fe = np.zeros((8, 2), dtype=np.float32)
    ret._texts = [f"passage {i}" for i in range(8)]

    routed_queries = []

    def fake_selector(query, top_k):
        routed_queries.append((query, top_k))
        return [0, 1, 2, 3]

    ret._retrieve_core = fake_selector
    result = ret.retrieve("Who won awards for Example Film?", top_k=3)

    assert [q for q, _ in routed_queries] == [
        "Who won awards for Example Film?",
        "Who directed Example Film?",
        "What awards did the director win? Jane Doe",
    ]
    assert all(k == 4 for _, k in routed_queries)
    assert result == [0, 1, 2]


def test_historical_decomposition_keeps_full_cosine_path():
    from membukkit.retrieval.multihop import CoreMemRetriever

    class FakeEncoder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype=np.float32)

    ret = CoreMemRetriever(
        encoder_path="unused",
        reranker_path="unused",
        decompose=True,
        decompose_retrieval="full_cosine",
        rerank_cap=3,
        llm_fn=lambda prompt: "",
    )
    ret._enc = FakeEncoder()
    ret._fe = np.array([[0.1, 0.0], [0.9, 0.0], [0.5, 0.0]], dtype=np.float32)
    ret._texts = ["a", "b", "c"]
    ret._retrieve_core = lambda *_: pytest.fail("historical path called bucket selector")

    assert ret.retrieve("query", top_k=2) == [1, 2]


def test_bucketed_decomposition_rejects_legacy_maxpool():
    from membukkit.retrieval.multihop import CoreMemRetriever

    with pytest.raises(ValueError, match="requires interleave"):
        CoreMemRetriever(
            encoder_path="unused",
            reranker_path="unused",
            decompose=True,
            decompose_retrieval="bucket",
            decompose_fuse="maxpool",
        )


def test_bucket_selector_invokes_router_reranker_and_rrf(monkeypatch):
    from membukkit.retrieval import buckets
    from membukkit.retrieval.multihop import CoreMemRetriever

    calls = {}

    class FakeEncoder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype=np.float32)

    class FakeReranker:
        def score(self, query, texts):
            calls["reranker"] = (query, list(texts))
            return np.array([0.2, 0.9, 0.1], dtype=np.float32)

    def fake_route(partition, query_emb, budget, record):
        calls["route"] = (partition, budget, record)
        return [0, 1, 2], {"scan_frac": 0.3}

    def fake_rrf(util, cos):
        calls["rrf"] = (util.tolist(), cos.tolist())
        return [1, 2, 0]

    monkeypatch.setattr(buckets, "route_topic", fake_route)
    monkeypatch.setattr(buckets, "rrf_order", fake_rrf)

    ret = CoreMemRetriever(
        encoder_path="unused",
        reranker_path="unused",
        budget=0.3,
        fusion="rrf",
        rerank_cap=3,
    )
    ret._enc = FakeEncoder()
    ret._reranker = FakeReranker()
    ret._partition = {"test": True}
    ret._fe = np.array([[0.3, 0.0], [0.8, 0.0], [0.5, 0.0]], dtype=np.float32)
    ret._texts = ["a", "b", "c"]

    assert ret._retrieve_core("query", top_k=2) == [1, 2]
    assert calls["route"] == ({"test": True}, 0.3, False)
    assert calls["reranker"] == ("query", ["a", "b", "c"])
    assert "rrf" in calls


def test_cuda_reranker_calls_are_serialized_across_retrieval_threads():
    from membukkit.retrieval.multihop import CoreMemRetriever

    class ConcurrentCallDetector:
        def __init__(self):
            self.guard = threading.Lock()
            self.active = 0
            self.max_active = 0

        def score(self, query, texts):
            with self.guard:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.01)
            with self.guard:
                self.active -= 1
            return np.zeros(len(texts), dtype=np.float32)

    ret = CoreMemRetriever(encoder_path="unused", reranker_path="unused")
    detector = ConcurrentCallDetector()
    ret._reranker = detector

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: ret._score_reranker("query", ["passage"]), range(16)))

    assert detector.max_active == 1
