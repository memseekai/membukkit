"""Tests for the storage backend seam (InMemoryBackend + MemorySystem plumbing)."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np

from membukkit.config import RetrievalConfig
from membukkit.storage.base import FactRecord, content_id
from membukkit.storage.memory import InMemoryBackend


class FakeEncoder:
    """Deterministic, dependency-free encoder that counts how many texts it embeds."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.encoded_batches = []  # list of the text-lists it was asked to encode

    def encode(self, texts, normalize: bool = True, show_progress: bool = False):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        self.encoded_batches.append(items)
        vecs = []
        for t in items:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(self.dim).astype(np.float32)
            if normalize:
                v = v / (np.linalg.norm(v) + 1e-8)
            vecs.append(v)
        out = np.vstack(vecs).astype(np.float32)
        return out[0] if single else out


def _facts(texts, ts=None):
    ts = ts or datetime(2024, 6, 1)
    return [FactRecord(text=t, timestamp=ts) for t in texts]


def test_content_id_is_stable_and_dedups():
    a = content_id("The user has a dog named Luna.")
    b = content_id("the user has a   dog named luna.")  # whitespace/case-insensitive
    assert a == b
    assert content_id("different") != a


def test_recurring_fact_on_a_later_date_is_kept():
    """The same fact stated in a later session is a new dated observation,
    not a duplicate — only same-date re-ingest dedups."""
    same_day = content_id("I went for a run", date=datetime(2024, 6, 1))
    same_day_again = content_id("i went for a  RUN", date=datetime(2024, 6, 1, 18, 30))
    later = content_id("I went for a run", date=datetime(2024, 9, 15))
    assert same_day == same_day_again  # idempotent re-ingest of the same session
    assert later != same_day  # recurrence kept

    be = InMemoryBackend(RetrievalConfig(), FakeEncoder())
    be.upsert_facts(_facts(["I went for a run"], ts=datetime(2024, 6, 1)))
    be.upsert_facts(_facts(["I went for a run"], ts=datetime(2024, 6, 1)))  # dedup
    be.upsert_facts(_facts(["I went for a run"], ts=datetime(2024, 9, 15)))  # kept
    assert be.count() == 2


def test_undated_facts_dedup_on_content_alone():
    assert content_id("likes tea") == content_id("likes  TEA")


_CAT_SESSION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I adopted a cat named Miso."},
    {"role": "assistant", "content": "That's wonderful! Cats are great."},
    {"role": "user", "content": "She is two years old."},
]


def _mem(retrieval, distiller=None):
    from membukkit.config import PromptConfig
    from membukkit.pipeline import MemorySystem

    class FakeReranker:
        def score(self, query, texts, batch_size: int = 64):
            return np.ones(len(texts), dtype=np.float32)

    return MemorySystem(
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        llm_fn=lambda p: "unused",
        retrieval=retrieval,
        prompts=PromptConfig.default(),
        distiller=distiller,
    )


def test_raw_ingest_skips_assistant_and_system_turns_when_union_off():
    """union=False (single-index) keeps only what the person said."""
    mem = _mem(RetrievalConfig(union=False, num_buckets=2, scan_budget=1.0, top_k=5))
    mem.ingest(sessions=[_CAT_SESSION], dates=["2024-06-01"])

    stored = mem._backend._texts
    assert "I adopted a cat named Miso." in stored
    assert "She is two years old." in stored
    assert all("helpful assistant" not in t and "wonderful" not in t for t in stored)


def test_union_ingest_keeps_all_turns_verbatim():
    """union=True stores every turn (all roles) in the verbatim lane — the raw
    bank the SOTA eval retrieves over."""
    mem = _mem(RetrievalConfig(union=True, num_buckets=2, scan_budget=1.0, top_k=5))
    mem.ingest(sessions=[_CAT_SESSION], dates=["2024-06-01"])

    verbatim = [t for t, k in zip(mem._backend._texts, mem._backend._kinds) if k == "verbatim"]
    assert "I adopted a cat named Miso." in verbatim
    assert "She is two years old." in verbatim
    assert any("helpful assistant" in t for t in verbatim)
    assert any("wonderful" in t for t in verbatim)
    # No distiller -> no atomic lane; union degrades to verbatim-only.
    assert mem._backend.count_kind("atomic") == 0


def test_upsert_dedups_and_embeds_only_new():
    enc = FakeEncoder()
    be = InMemoryBackend(RetrievalConfig(), enc)

    n1 = be.upsert_facts(_facts(["fact one", "fact two"]))
    assert n1 == 2
    assert be.count() == 2
    assert enc.encoded_batches[-1] == ["fact one", "fact two"]

    # Re-ingest one existing + one new: only the NEW one is embedded.
    n2 = be.upsert_facts(_facts(["fact one", "fact three"]))
    assert n2 == 1
    assert be.count() == 3
    assert enc.encoded_batches[-1] == ["fact three"]


def test_candidates_returns_pool_with_trace():
    enc = FakeEncoder()
    cfg = RetrievalConfig(num_buckets=4, scan_budget=0.5)
    be = InMemoryBackend(cfg, enc)
    be.upsert_facts(_facts([f"fact number {i}" for i in range(30)]))

    pool = be.candidates("fact number 3", top_k=10)
    assert len(pool.candidates) > 0
    assert pool.has_cosine is True  # topic + hybrid path computes cosines
    assert pool.trace["scan_frac"] > 0
    assert pool.trace["n_facts"] == 30


def test_clear_resets_state():
    be = InMemoryBackend(RetrievalConfig(), FakeEncoder())
    be.upsert_facts(_facts(["a", "b"]))
    assert be.count() == 2
    be.clear()
    assert be.count() == 0
    assert be.candidates("a", top_k=5).candidates == []


def test_memorysystem_end_to_end_with_fakes():
    from membukkit.pipeline import MemorySystem

    enc = FakeEncoder()

    class FakeReranker:
        def score(self, query, texts, batch_size: int = 64):
            # Prefer texts that share words with the query.
            qs = set(query.lower().split())
            return np.asarray([len(qs & set(t.lower().split())) for t in texts], dtype=np.float32)

    captured = {}

    def fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "answered"

    mem = MemorySystem(
        encoder=enc,
        reranker=FakeReranker(),
        llm_fn=fake_llm,
        retrieval=RetrievalConfig(num_buckets=4, scan_budget=0.6, top_k=5),
        prompts=__import__("membukkit.config", fromlist=["PromptConfig"]).PromptConfig.default(),
        distiller=None,
    )
    sessions = [[{"role": "user", "content": f"I like activity {i}"}] for i in range(12)]
    mem.ingest(sessions=sessions, dates=["2024/06/0{}".format((i % 9) + 1) for i in range(12)])
    assert mem._backend.count() == 12

    res = mem.answer("which activity 3 do I like")
    assert res.answer  # non-empty
    assert res.facts  # retrieved some facts
    assert res.trace.n_facts == 12

    # reset clears everything
    mem.reset()
    assert mem._backend.count() == 0


def test_memorysystem_answer_can_skip_reader_llm():
    """generate_answer=False returns facts without ever invoking the reader LLM."""
    from membukkit.pipeline import MemorySystem
    from membukkit.config import PromptConfig

    enc = FakeEncoder()

    class FakeReranker:
        def score(self, query, texts, batch_size: int = 64):
            qs = set(query.lower().split())
            return np.asarray([len(qs & set(t.lower().split())) for t in texts], dtype=np.float32)

    def exploding_llm(prompt: str) -> str:
        raise AssertionError("reader LLM must not be called when generate_answer=False")

    mem = MemorySystem(
        encoder=enc,
        reranker=FakeReranker(),
        llm_fn=exploding_llm,
        retrieval=RetrievalConfig(num_buckets=4, scan_budget=0.6, top_k=5),
        prompts=PromptConfig.default(),
        distiller=None,
    )
    sessions = [[{"role": "user", "content": f"I like activity {i}"}] for i in range(12)]
    mem.ingest(sessions=sessions, dates=["2024/06/0{}".format((i % 9) + 1) for i in range(12)])

    res = mem.answer("which activity 3 do I like", generate_answer=False)
    assert res.answer is None  # reader skipped
    assert res.facts  # retrieval still ran
    assert res.trace.n_facts == 12  # trace still computed


def test_memorysystem_search_returns_citable_evidence():
    from membukkit.pipeline import MemorySystem
    from membukkit.config import PromptConfig

    enc = FakeEncoder()

    class FakeReranker:
        def score(self, query, texts, batch_size: int = 64):
            qs = set(query.lower().split())
            return np.asarray([len(qs & set(t.lower().split())) for t in texts], dtype=np.float32)

    mem = MemorySystem(
        encoder=enc,
        reranker=FakeReranker(),
        llm_fn=lambda p: "unused",
        retrieval=RetrievalConfig(num_buckets=4, scan_budget=1.0, top_k=3),
        prompts=PromptConfig.default(),
        distiller=None,
    )
    mem.ingest(
        sessions=[
            [{"role": "user", "content": "I meet Dana in Lisbon on Tuesday."}],
            [{"role": "user", "content": "I prefer morning strategy meetings."}],
            [{"role": "user", "content": "My dentist is on Friday."}],
        ],
        dates=["2024/06/01", "2024/06/02", "2024/06/03"],
    )

    res = mem.search("Dana Lisbon meeting", top_k=2)
    assert res.query == "Dana Lisbon meeting"
    assert len(res.hits) == 2
    assert all(hit.ref.startswith("mem:") for hit in res.hits)
    assert all(hit.fact.startswith("[2024-06-") for hit in res.hits)
    assert res.trace.reader_type == "search"
    assert res.trace.ranked_facts == [hit.fact for hit in res.hits]


def test_memorysystem_accepts_iso_dates_and_traces_fact_times():
    from membukkit.pipeline import MemorySystem
    from membukkit.config import PromptConfig

    enc = FakeEncoder()

    class FakeReranker:
        def score(self, query, texts, batch_size: int = 64):
            return np.ones(len(texts), dtype=np.float32)

    captured = {}

    def fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "answered"

    mem = MemorySystem(
        encoder=enc,
        reranker=FakeReranker(),
        llm_fn=fake_llm,
        retrieval=RetrievalConfig(num_buckets=2, scan_budget=1.0, top_k=2),
        prompts=PromptConfig.default(),
        distiller=None,
    )
    mem.ingest(
        sessions=[
            [{"role": "user", "content": "I have an ISO dated fact."}],
            [{"role": "user", "content": "I have an offset dated fact."}],
        ],
        dates=["2024-06-01T10:30:00", "2024-06-02T12:00:00-05:00"],
    )

    res = mem.answer("what facts do I have?", question_date="2024-07-01T09:00:00Z")
    assert all(line.startswith("[2024-06-") for line in res.facts)
    assert res.trace.ranked_fact_times == [
        "2024-06-01T10:30:00",
        "2024-06-02T12:00:00-05:00",
    ]
    assert "Today's date is 2024-07-01T09:00:00+00:00." in captured["prompt"]


def test_present_temporal_sorts_mixed_naive_and_aware_timestamps():
    from membukkit.pipeline import MemorySystem

    cands = [
        SimpleNamespace(
            text="aware late", timestamp=datetime.fromisoformat("2024-06-01T09:00:00-05:00")
        ),
        SimpleNamespace(text="naive early", timestamp=datetime(2024, 6, 1, 8, 0, 0)),
    ]
    lines, times = MemorySystem._present_temporal_with_times(cands)
    assert lines == ["[2024-06-01] naive early", "[2024-06-01] aware late"]
    assert times == ["2024-06-01T08:00:00", "2024-06-01T09:00:00-05:00"]
