"""Basic smoke tests for MEMBUKKIT library."""

from __future__ import annotations

import pytest
import numpy as np


def test_imports():
    from membukkit import MemorySystem, RetrievalConfig, ModelConfig, PromptConfig

    assert MemorySystem is not None
    assert RetrievalConfig is not None


def test_config_defaults():
    from membukkit.config import RetrievalConfig, ModelConfig, PromptConfig

    rc = RetrievalConfig()
    assert rc.num_buckets == 24
    assert rc.scan_budget == 0.3
    assert rc.select == "hybrid"
    mc = ModelConfig()
    assert mc.encoder == "biencoder_v1"
    pc = PromptConfig.default()
    assert pc.extraction is None


def test_rrf_order():
    from membukkit.retrieval.buckets import rrf_order

    util = np.array([0.9, 0.1, 0.5, 0.3])
    cos = np.array([0.1, 0.9, 0.3, 0.5])
    order = rrf_order(util, cos)
    assert len(order) == 4
    assert all(i in order for i in range(4))


def test_build_topic_partition():
    from membukkit.retrieval.buckets import build_topic_partition

    rng = np.random.default_rng(42)
    embs = rng.standard_normal((50, 64)).astype(np.float32)
    part = build_topic_partition(embs, k=5, seed=42)
    assert "labels" in part
    assert "by_bucket" in part
    assert part["n"] == 50
    assert part["k_eff"] <= 5


def test_route_topic():
    from membukkit.retrieval.buckets import build_topic_partition, route_topic

    rng = np.random.default_rng(42)
    embs = rng.standard_normal((50, 64)).astype(np.float32)
    part = build_topic_partition(embs, k=5, seed=42)
    query = rng.standard_normal(64).astype(np.float32)
    cand, trace = route_topic(part, query, budget=0.3)
    assert len(cand) > 0
    assert trace["scan_frac"] > 0


def test_query_router():
    from membukkit.retrieval.router import QueryRouter, QueryClass

    router = QueryRouter()
    d = router.route("When did I first visit Paris?")
    assert d.query_class in (QueryClass.TEMPORAL, QueryClass.KNOWLEDGE_UPDATE)
    d2 = router.route("What is my dog's name?")
    assert d2.query_class == QueryClass.GENERAL


def test_is_recommendation():
    from membukkit.retrieval.router import is_recommendation_query

    assert is_recommendation_query("Can you recommend a good restaurant?")
    assert not is_recommendation_query("What is my dog's name?")


def test_fact_distiller_parse():
    from membukkit.extraction.distiller import parse_facts

    raw = "0 | The user has a dog named Luna\n1 | The user lives in Portland"
    facts = parse_facts(raw)
    assert len(facts) == 2
    assert facts[0][0] == 0
    assert "Luna" in facts[0][1]


def test_build_transcript():
    from membukkit.extraction.distiller import build_transcript

    turns = [("user", "Hello"), ("assistant", "Hi there")]
    t = build_transcript(turns, numbered=True)
    assert "[T0]" in t
    assert "Hello" in t


def test_f1_score():
    from membukkit.eval.metrics import f1_score

    assert f1_score("the dog is named Luna", "Luna the dog") > 0.5
    assert f1_score("completely wrong", "Luna the dog") == 0.0
    assert f1_score("", "") == 1.0


def test_data_types():
    from datetime import datetime

    from membukkit.data.base import FactInput, QueryInput

    f = FactInput(text="test fact", timestamp=datetime(2024, 1, 1))
    assert f.text == "test fact"
    q = QueryInput(text="test question")
    assert q.text == "test question"


def test_extract_entities():
    from membukkit.retrieval.bucket_index import extract_entities

    ents = extract_entities("I visited Paris with my brother John")
    assert "paris" in ents
    assert "john" in ents
    assert "brother" in ents


def test_llm_backend_parse():
    from membukkit.llm.backends import parse_llm_spec

    backend = parse_llm_spec("openai:gpt-4o-mini")
    assert backend is not None


def test_model_registry():
    from membukkit.models.registry import resolve_encoder_path, resolve_reranker_path
    from membukkit.config import ModelConfig

    mc = ModelConfig(model_dir="/tmp/test_models")
    enc = resolve_encoder_path(mc)
    assert "biencoder_v1" in enc
    rer = resolve_reranker_path(mc)
    assert rer.endswith("/model")
