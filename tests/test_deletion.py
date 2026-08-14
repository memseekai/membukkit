"""User-initiated erasure.

MemBukkit's model is append-and-supersede: updates keep the old fact so as-of
answers work. Deletion is the deliberate exception, for facts that are wrong or
that the user does not want retained. These tests pin the three properties that
make it honest rather than cosmetic:

1. deleting an atomic fact takes its verbatim source with it, so the content is
   not still sitting in the other lane,
2. a fact superseded by a deleted fact becomes current again,
3. a source still needed by a surviving fact is kept.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from membukkit.config import RetrievalConfig
from membukkit.storage.base import FactRecord
from membukkit.storage.memory import InMemoryBackend
from membukkit.supersession import is_active_as_of


class FakeEncoder:
    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode(self, texts, normalize: bool = True, show_progress: bool = False):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vecs = []
        for t in items:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(self.dim).astype(np.float32)
            vecs.append(v / (np.linalg.norm(v) + 1e-8))
        out = np.vstack(vecs).astype(np.float32)
        return out[0] if single else out


def _rec(text, *, kind, ref, doc="d1", session="s1", ts=datetime(2024, 6, 1)):
    return FactRecord(
        text=text, timestamp=ts, kind=kind, doc_id=doc,
        source_session=session, source_ref=ref,
    )


def _backend(records):
    be = InMemoryBackend(RetrievalConfig(), FakeEncoder())
    be.upsert_facts(records)
    return be


def _id_of(be, text):
    return be._ids[be._texts.index(text)]


def _texts(be):
    return list(be._texts)


# ------------------------------------------------- source-linked erasure
def test_deleting_a_fact_removes_its_verbatim_source():
    """Otherwise the content the user deleted is still retrievable."""
    be = _backend([
        _rec("Rent is 800 EUR.", kind="atomic", ref="session:0/turn:0"),
        _rec("my rent is 800 eur now", kind="verbatim", ref="session:0/turn:0"),
        _rec("Gym is on Tuesdays.", kind="atomic", ref="session:0/turn:1"),
        _rec("i go to the gym tuesdays", kind="verbatim", ref="session:0/turn:1"),
    ])
    fid = _id_of(be, "Rent is 800 EUR.")

    extra = be.orphaned_source_ids([fid])
    assert len(extra) == 1
    assert be.delete_facts([fid, *extra]) == 2

    remaining = _texts(be)
    assert not any("800" in t for t in remaining), "rent content survived deletion"
    assert "Gym is on Tuesdays." in remaining
    assert "i go to the gym tuesdays" in remaining


def test_shared_source_is_kept_for_surviving_facts():
    """One turn can distil into several facts; the source outlives any one."""
    be = _backend([
        _rec("Rent is 800 EUR.", kind="atomic", ref="session:0/turn:0"),
        _rec("Groceries are 300 EUR.", kind="atomic", ref="session:0/turn:0"),
        _rec("budget: 800 rent, 300 groceries", kind="verbatim", ref="session:0/turn:0"),
    ])
    assert be.orphaned_source_ids([_id_of(be, "Rent is 800 EUR.")]) == []

    # once both facts go, the source is orphaned and removed
    both = [_id_of(be, "Rent is 800 EUR."), _id_of(be, "Groceries are 300 EUR.")]
    assert len(be.orphaned_source_ids(both)) == 1


def test_purge_source_removes_everything_from_the_turn():
    be = _backend([
        _rec("Rent is 800 EUR.", kind="atomic", ref="session:0/turn:0"),
        _rec("Groceries are 300 EUR.", kind="atomic", ref="session:0/turn:0"),
        _rec("budget: 800 rent, 300 groceries", kind="verbatim", ref="session:0/turn:0"),
        _rec("Gym is on Tuesdays.", kind="atomic", ref="session:0/turn:1"),
    ])
    ids = be.source_group_ids([_id_of(be, "Rent is 800 EUR.")])
    be.delete_facts(ids)
    assert _texts(be) == ["Gym is on Tuesdays."]


# ------------------------------------------------------ supersession repair
def test_deleting_a_correction_revives_what_it_superseded():
    """The headline case: the *new* fact was wrong, so removing it must not
    leave the store with no current value at all."""
    be = _backend([
        _rec("Rent is 800 EUR.", kind="atomic", ref="session:0/turn:0",
             ts=datetime(2024, 1, 8)),
        _rec("Rent is 950 EUR.", kind="atomic", ref="session:1/turn:0",
             ts=datetime(2024, 4, 2)),
    ])
    old, new = _id_of(be, "Rent is 800 EUR."), _id_of(be, "Rent is 950 EUR.")
    be.supersede([(old, new)], when=datetime(2024, 4, 2))

    i = be._ids.index(old)
    assert be._superseded_by[i] == new
    assert not is_active_as_of(
        superseded_by=be._superseded_by[i], valid_to=be._valid_to[i],
        timestamp=be._times[i], as_of=datetime(2024, 6, 1),
    )

    be.delete_facts([new])

    i = be._ids.index(old)
    assert be._superseded_by[i] == "", "dangling supersession pointer left behind"
    assert be._valid_to[i] is None
    assert is_active_as_of(
        superseded_by=be._superseded_by[i], valid_to=be._valid_to[i],
        timestamp=be._times[i], as_of=datetime(2024, 6, 1),
    ), "the surviving fact must be current again"


def test_unrelated_supersession_links_are_untouched():
    be = _backend([
        _rec("Job at Acme.", kind="atomic", ref="session:0/turn:0"),
        _rec("Job at Globex.", kind="atomic", ref="session:1/turn:0"),
        _rec("Gym is on Tuesdays.", kind="atomic", ref="session:2/turn:0"),
    ])
    old, new = _id_of(be, "Job at Acme."), _id_of(be, "Job at Globex.")
    be.supersede([(old, new)], when=datetime(2024, 4, 2))

    be.delete_facts([_id_of(be, "Gym is on Tuesdays.")])
    assert be._superseded_by[be._ids.index(old)] == new


# ------------------------------------------------------------- bulk + caches
def test_ids_for_source_scopes_to_document_and_session():
    be = _backend([
        _rec("A.", kind="atomic", ref="session:0/turn:0", doc="d1", session="s1"),
        _rec("B.", kind="verbatim", ref="session:0/turn:0", doc="d1", session="s1"),
        _rec("C.", kind="atomic", ref="session:0/turn:0", doc="d2", session="s2"),
    ])
    assert len(be.ids_for_source(doc_id="d1")) == 2
    assert len(be.ids_for_source(source_session="s2")) == 1
    assert be.ids_for_source(doc_id="nope") == []


def test_delete_invalidates_the_lexical_index():
    """Row positions shift on delete, so a cached BM25 index must not survive."""
    pytest.importorskip("rank_bm25")
    be = InMemoryBackend(RetrievalConfig(lexical_lane=True), FakeEncoder())
    be.upsert_facts([
        _rec("alpha unique-token", kind="atomic", ref="session:0/turn:0"),
        _rec("beta filler text", kind="atomic", ref="session:0/turn:1"),
        _rec("gamma filler text", kind="atomic", ref="session:0/turn:2"),
    ])
    be.candidates("unique-token", top_k=3)  # populates the cache
    assert be._bm25

    be.delete_facts([_id_of(be, "alpha unique-token")])
    assert be._bm25 == {}, "stale lexical index survived a delete"

    pool = be.candidates("unique-token", top_k=3)
    assert not any("alpha" in c.text for c in pool.candidates)


def test_ingest_provenance_is_unique_per_session():
    """Two unrelated ingests must not claim the same source.

    They used to: `session_id` was `ingest:{index-in-this-call}`, so every
    standalone `add` wrote `ingest:0`. Erasing a source then matched unrelated
    rows, and `--purge-source` deleted the entire store.
    """
    from membukkit.config import PromptConfig
    from membukkit.pipeline import MemorySystem

    class NoopDistiller:
        def distill(self, key, transcript, date_str, mode=None):
            return []

    mem = MemorySystem(
        encoder=FakeEncoder(),
        reranker=None,
        llm_fn=lambda _p: "",
        retrieval=RetrievalConfig(),
        prompts=PromptConfig.default(),
        distiller=NoopDistiller(),
    )
    mem.ingest(sessions=[[{"role": "user", "content": "Rent is 800."}]], dates=["2024-01-08"])
    mem.ingest(sessions=[[{"role": "user", "content": "Gym on Tuesdays."}]], dates=["2024-02-01"])

    sources = set(mem.backend._sources)
    assert len(sources) == 2, f"separate ingests shared provenance: {sources}"
    assert all(s.startswith("ingest:0:") for s in sources)

    # and erasing one source leaves the other intact
    lease = [
        f for i, f in enumerate(mem.backend._ids) if "800" in mem.backend._texts[i]
    ]
    mem.delete_facts(lease, purge_source=True)
    assert any("Gym" in t for t in mem.backend._texts)


def test_delete_is_a_no_op_for_unknown_ids():
    be = _backend([_rec("A.", kind="atomic", ref="session:0/turn:0")])
    assert be.delete_facts(["nope"]) == 0
    assert be.orphaned_source_ids(["nope"]) == []
    assert be.count() == 1
