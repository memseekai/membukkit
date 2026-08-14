"""Tests for the optional BM25 lexical lane (`RetrievalConfig.lexical_lane`).

Two things must hold: with the flag off the retrieval path is exactly what it
was before the lane existed (no import, no extra candidates, no score fields
set), and with it on a fact that dense routing never surfaces can still reach
the pool on an exact term match.
"""

from __future__ import annotations

import sys
from datetime import datetime

import numpy as np
import pytest

from membukkit.config import RetrievalConfig
from membukkit.retrieval.buckets import rrf_order
from membukkit.retrieval.lexical import BM25Index, available, tokenize
from membukkit.storage.base import FactRecord
from membukkit.storage.memory import InMemoryBackend

# The lane is opt-in, so `rank_bm25` may legitimately be absent. Everything that
# actually retrieves lexically skips without it; the off-by-default tests below
# must run either way, since they are what guarantee the dense path is untouched.
requires_bm25 = pytest.mark.skipif(not available(), reason="needs the `bm25` extra")

RARE = "zzqx4417"
# Present only in the outlier fact, never in a query, so the encoder below can
# strand that fact in embedding space while BM25 can still match it on RARE.
MARKER = "ingress"


class SplitEncoder:
    """Deterministic encoder with two orthogonal clusters.

    Text containing MARKER embeds onto axis 1; everything else (every filler
    fact and every query in this file) embeds onto axis 0 with a small
    per-text jitter so KMeans can still form buckets. The outlier fact is
    therefore unreachable by dense routing however the query is worded, which
    is exactly the case the lexical lane exists for.
    """

    def __init__(self, dim: int = 16):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        if MARKER in text.lower():
            v[1] = 1.0
            return v
        v[0] = 1.0
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v[2:] = rng.standard_normal(self.dim - 2).astype(np.float32) * 0.01
        return v / (np.linalg.norm(v) + 1e-8)

    def encode(self, texts, normalize: bool = True, show_progress: bool = False):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        out = np.vstack([self._vec(t) for t in items]).astype(np.float32)
        return out[0] if single else out


def _backend(cfg: RetrievalConfig) -> InMemoryBackend:
    be = InMemoryBackend(cfg, SplitEncoder())
    facts = [
        FactRecord(text=f"Standup notes for sprint {i} covering the payments work.",
                   timestamp=datetime(2024, 6, 1), kind="atomic")
        for i in range(40)
    ]
    facts.append(
        FactRecord(
            text=f"The staging deploy failed with error code {RARE} on the ingress.",
            timestamp=datetime(2024, 6, 2),
            kind="atomic",
        )
    )
    be.upsert_facts(facts)
    return be


def _texts(pool) -> list:
    return [c.text for c in pool.candidates]


# ----------------------------------------------------------------- unit bits
def test_tokenize_keeps_snake_case_and_lowercases():
    assert tokenize("Deploy_Failed with ERROR-42!") == ["deploy_failed", "with", "error", "42"]


@requires_bm25
def test_bm25_index_ranks_exact_term_first_and_drops_non_matches():
    idx = BM25Index(["the cat sat", f"deploy failed {RARE}", "unrelated text"])
    hits = idx.top_k(RARE, 5)
    assert hits, "exact term should match"
    assert hits[0][0] == 1
    # a document sharing no query term is not a lexical hit at all
    assert all(pos != 2 for pos, _ in hits)


@requires_bm25
def test_bm25_index_empty_query_returns_nothing():
    idx = BM25Index(["alpha beta"])
    assert idx.top_k("", 5) == []
    assert idx.top_k("!!!", 5) == []


def test_rrf_order_two_signal_form_is_unchanged():
    util = [0.9, 0.1, 0.5]
    cos = [0.1, 0.9, 0.5]
    assert rrf_order(util, cos) == rrf_order(util, cos, k_rrf=60)
    assert sorted(rrf_order(util, cos)) == [0, 1, 2]


def test_rrf_order_third_signal_lifts_its_top_item():
    util = [0.1, 0.1, 0.1]
    cos = [0.1, 0.1, 0.1]
    lex = [0.0, 0.0, 9.0]
    assert rrf_order(util, cos, lex)[0] == 2


# ------------------------------------------------------------------ lane off
def test_lane_off_by_default():
    assert RetrievalConfig().lexical_lane is False


def test_lane_off_leaves_pool_and_scores_untouched():
    be = _backend(RetrievalConfig())
    pool = be.candidates("payments sprint standup", top_k=10)
    assert pool.has_lexical is False
    assert all(c.lexical == 0.0 for c in pool.candidates)
    assert "lexical_added" not in pool.trace
    # the rare fact is orthogonal to the query, so dense routing never sees it
    assert not any(RARE in t.lower() for t in _texts(pool))


def test_lane_off_does_not_import_rank_bm25(monkeypatch):
    """The optional dependency must not be touched on the default path."""
    monkeypatch.delitem(sys.modules, "rank_bm25", raising=False)
    be = _backend(RetrievalConfig())
    be.candidates("payments sprint standup", top_k=10)
    assert "rank_bm25" not in sys.modules


@requires_bm25
def test_lane_off_matches_pool_with_lane_never_configured():
    """Turning the flag on and back off yields the identical routed pool."""
    query = "payments sprint standup"
    base = _backend(RetrievalConfig())
    baseline = _texts(base.candidates(query, top_k=10))

    on = _backend(RetrievalConfig(lexical_lane=True))
    on.candidates(query, top_k=10)  # builds and caches the BM25 index
    on._cfg = RetrievalConfig()  # flag back off
    assert _texts(on.candidates(query, top_k=10)) == baseline


# ------------------------------------------------------------------- lane on
@requires_bm25
def test_lane_on_admits_a_fact_dense_routing_missed():
    cfg = RetrievalConfig(lexical_lane=True)
    be = _backend(cfg)
    pool = be.candidates(f"what was error code {RARE}", top_k=10)
    assert pool.has_lexical is True
    assert any(RARE in t.lower() for t in _texts(pool)), "BM25 should admit the exact match"
    assert pool.trace["lexical_added"] >= 1
    assert pool.trace["lexical_scanned"] == 41
    # the routed candidates are still present; the lane adds, never removes
    assert len(pool.candidates) > pool.trace["lexical_added"]


@requires_bm25
def test_lane_on_scores_only_lexical_hits():
    be = _backend(RetrievalConfig(lexical_lane=True))
    pool = be.candidates(f"what was error code {RARE}", top_k=10)
    scored = [c for c in pool.candidates if c.lexical > 0.0]
    assert scored, "at least the exact match carries a BM25 score"
    assert all(RARE in c.text.lower() or "error" in c.text.lower() for c in scored)


@requires_bm25
def test_lane_on_kind_scoped_path():
    """The union path retrieves per kind; the lane must work there too."""
    be = _backend(RetrievalConfig(lexical_lane=True))
    pool = be.candidates(f"error code {RARE}", top_k=10, kind="atomic")
    assert pool.has_lexical is True
    assert any(RARE in t.lower() for t in _texts(pool))


@requires_bm25
def test_lane_on_with_no_term_overlap_is_a_no_op():
    be = _backend(RetrievalConfig(lexical_lane=True))
    pool = be.candidates("!!! ???", top_k=10)
    assert pool.has_lexical is False
    assert "lexical_added" not in pool.trace


@requires_bm25
def test_index_rebuilds_when_facts_are_appended():
    cfg = RetrievalConfig(lexical_lane=True)
    be = _backend(cfg)
    be.candidates("payments", top_k=5)  # build cache at 41 rows
    be.upsert_facts(
        [FactRecord(text="A brand new hotfix note about tokens.", timestamp=datetime(2024, 7, 1))]
    )
    pool = be.candidates("hotfix tokens", top_k=5)
    assert any("hotfix" in t.lower() for t in _texts(pool))
    assert pool.trace["lexical_scanned"] == 42


def test_missing_dependency_raises_with_install_hint(monkeypatch):
    """When the extra is not installed, the error must say how to fix it."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "rank_bm25":
            raise ImportError("No module named 'rank_bm25'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "rank_bm25", raising=False)
    with pytest.raises(ImportError, match=r'membukkit\[bm25\]'):
        BM25Index(["some text"])
