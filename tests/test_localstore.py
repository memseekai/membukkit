"""LocalStore round trip: persist a backend, reload it, resolve provenance."""

from datetime import datetime

import numpy as np
import pytest

from membukkit.config import RetrievalConfig
from membukkit.storage.base import FactRecord
from membukkit.storage.localstore import LocalStore, list_stores
from membukkit.storage.memory import InMemoryBackend


class FakeEncoder:
    dim = 8

    def encode(self, texts, normalize=True, show_progress=False):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        rng = [abs(hash(t)) % 997 for t in items]
        vecs = np.stack([np.linspace(r, r + 1, self.dim) for r in rng]).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs[0] if single else vecs


@pytest.fixture()
def store_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMBUKKIT_HOME", str(tmp_path))
    return tmp_path


def _backend_with_facts():
    backend = InMemoryBackend(RetrievalConfig(union=True), FakeEncoder())
    backend.upsert_facts(
        [
            FactRecord(
                text="I adopted a golden retriever named Biscuit",
                timestamp=datetime(2024, 3, 1),
                kind="atomic",
                source_session="ingest:0",
                doc_id="doc123",
                doc_name="chat_export.json",
                source_ref="session:0",
            ),
            FactRecord(
                text="user: my dog Biscuit chewed the couch again",
                timestamp=datetime(2024, 4, 2),
                kind="verbatim",
                source_session="ingest:1",
                source_speaker="user",
                doc_id="doc123",
                doc_name="chat_export.json",
                source_ref="session:1/turn:0",
            ),
        ]
    )
    return backend


def test_save_load_round_trip(store_home):
    backend = _backend_with_facts()
    store = LocalStore("pets")
    store.save_backend(backend)

    fresh = InMemoryBackend(RetrievalConfig(union=True), FakeEncoder())
    n = store.load_backend(fresh)
    assert n == 2
    assert fresh.count() == 2
    assert fresh.count_kind("verbatim") == 1
    assert fresh.count_kind("atomic") == 1

    # provenance survived
    fact = fresh.facts_page(limit=10)["facts"][0]
    assert fact["doc_id"] == "doc123"
    assert fact["doc_name"] == "chat_export.json"

    # embeddings survived and retrieval works on the reloaded bank
    pool = fresh.candidates("dog", top_k=5, kind="verbatim")
    assert pool.candidates
    assert pool.candidates[0].doc_id == "doc123"

    # idempotent re-save
    store.save_backend(fresh)
    assert store.meta()["n_facts"] == 2


def test_store_listing_and_meta(store_home):
    LocalStore("alpha")
    LocalStore("beta")
    names = [s["name"] for s in list_stores()]
    assert names == ["alpha", "beta"]


def test_document_registry_and_source_resolution(store_home):
    store = LocalStore("docs")
    sessions = [
        [
            {"role": "user", "content": "How do I renew my passport?"},
            {"role": "assistant", "content": "You can renew online via the portal."},
        ],
        [
            {"role": "user", "content": "The portal rejected my photo."},
            {"role": "assistant", "content": "Photos must be 35x45mm on white."},
        ],
    ]
    doc_id = store.add_document("support_tickets.json", sessions, dates=["2024-01-01", "2024-02-02"])

    docs = store.documents()
    assert len(docs) == 1 and docs[0]["doc_id"] == doc_id

    src = store.resolve_source(doc_id, "session:1/turn:0", context=1)
    assert src["session"] == 1
    assert src["date"] == "2024-02-02"
    assert src["turns"][src["highlight"]]["content"] == "The portal rejected my photo."

    whole = store.resolve_source(doc_id, "session:0")
    assert len(whole["turns"]) == 2 and whole["highlight"] is None

    # Stored turn-level refs report highlight_kind="stored".
    stored = store.resolve_source(doc_id, "session:1/turn:0", context=1)
    assert stored["highlight_kind"] == "stored"

    # Session-only refs + fact_text fall back to lexical turn attribution:
    # the fact's distinctive words pick out the matching turn.
    lex = store.resolve_source(
        doc_id, "session:1", fact_text="The user's photo was rejected by the portal."
    )
    assert lex["highlight_kind"] == "lexical"
    assert lex["turns"][lex["highlight"]]["content"] == "The portal rejected my photo."

    # No overlap at all -> whole session, no highlight.
    none = store.resolve_source(doc_id, "session:1", fact_text="zzz qqq xyzzy")
    assert none["highlight"] is None and none["highlight_kind"] is None


def test_best_matching_turn_weighs_rare_tokens(store_home):
    from membukkit.storage.localstore import best_matching_turn

    turns = [
        {"role": "user", "content": "We discussed the budget for the project today."},
        {"role": "user", "content": "The budget was raised to 900 euros for Lisbon."},
        {"role": "user", "content": "Weather is nice today."},
    ]
    # "budget" appears in two turns (low weight); "lisbon"/"900" are unique.
    idx = best_matching_turn("The user's Lisbon budget is 900 euros.", turns)
    assert idx == 1
    assert best_matching_turn("", turns) is None
    assert best_matching_turn("anything", []) is None


def test_bad_store_name(store_home):
    with pytest.raises(ValueError):
        LocalStore("../evil")
    with pytest.raises(FileNotFoundError):
        LocalStore("missing", create=False)


def test_get_fact_lookup(store_home):
    backend = _backend_with_facts()
    page = backend.facts_page(limit=1)
    fid = page["facts"][0]["id"]
    fact = backend.get_fact(fid)
    assert fact is not None and fact["id"] == fid
    assert backend.get_fact("nonexistent") is None
