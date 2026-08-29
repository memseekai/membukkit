"""Tests for the RAG document retriever.

Models are injected as deterministic fakes so this runs in CI with no weights
downloaded. The fakes are bag-of-words, which is enough to exercise the thing
that actually matters here: whether a second-hop document that shares no
vocabulary with the original question can still be retrieved.
"""

from __future__ import annotations

import re
import zlib

import numpy as np
import pytest

from membukkit.retrieval.rag import Document, Hit, RagRetriever, _interleave

_TOKEN = re.compile(r"[a-z0-9']+")
_DIM = 96


def _tokens(text: str):
    return _TOKEN.findall(text.lower())


class FakeEncoder:
    """Hashing bag-of-words. Cosine similarity is then token overlap.

    crc32, not hash(): Python randomises string hashing per process, so hash()
    would make which tokens collide differ run to run and these tests flaky.
    """

    def encode(self, texts, normalize: bool = True, show_progress: bool = False):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        out = np.zeros((len(items), _DIM))
        for row, text in enumerate(items):
            for tok in _tokens(text):
                out[row, zlib.crc32(tok.encode()) % _DIM] += 1.0
        if normalize:
            n = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.where(n == 0.0, 1.0, n)
        return out[0] if single else out


class FakeReranker:
    """Utility = count of query tokens present in the document."""

    def score(self, query: str, texts, batch_size: int = 64):
        q = set(_tokens(query))
        return np.asarray([float(len(q & set(_tokens(t)))) for t in texts])


# A synthetic bridge question: doc B is reachable only through doc A's entities.
QUERY = "What award did the director of Film X win?"
BRIDGE_DOCS = [
    Document("a", "Film X is a 1998 movie. Film X was directed by Jane Roe.", "Film X"),
    Document("b", "Jane Roe is a filmmaker. Jane Roe won the Palme dOr.", "Jane Roe"),
    Document("c", "Cooking with lentils requires patience and stock.", "Lentils"),
    Document("d", "The Baltic Sea is bordered by nine countries.", "Baltic Sea"),
    Document("e", "Zebras display distinctive black and white striping.", "Zebras"),
]


def build(mode="chain", **kw):
    return RagRetriever(mode=mode, encoder=FakeEncoder(), reranker=FakeReranker(), **kw)


# ----------------------------------------------------------------- indexing
def test_index_returns_document_count():
    r = build()
    assert r.index(BRIDGE_DOCS) == 5


def test_index_rejects_duplicate_doc_ids():
    r = build()
    with pytest.raises(ValueError, match="duplicate"):
        r.index([Document("x", "one"), Document("x", "two")])


def test_index_rejects_empty_doc_id():
    r = build()
    with pytest.raises(ValueError, match="doc_id"):
        r.index([Document("", "text")])


def test_reindex_replaces_previous_corpus():
    r = build()
    r.index(BRIDGE_DOCS)
    r.index([Document("only", "Jane Roe won the Palme dOr.")])
    assert [h.doc_id for h in r.search(QUERY)] == ["only"]


def test_search_on_empty_index_returns_empty():
    assert build().search("anything") == []


def test_index_empty_is_allowed():
    assert build().index([]) == 0


# ------------------------------------------------------------------- modes
def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="mode must be one of"):
        RagRetriever(mode="telepathy")


def test_decompose_without_llm_rejected():
    with pytest.raises(ValueError, match="requires an llm"):
        RagRetriever(mode="decompose")


def test_dense_finds_the_lexically_similar_document():
    r = build("dense")
    r.index(BRIDGE_DOCS)
    assert r.search(QUERY, top_k=1)[0].doc_id == "a"


def test_dense_misses_the_second_hop():
    """The premise of the whole module: one query cannot reach doc b."""
    r = build("dense")
    r.index(BRIDGE_DOCS)
    assert "b" not in [h.doc_id for h in r.search(QUERY, top_k=2)]


def test_chain_recovers_the_second_hop():
    r = build("chain")
    r.index(BRIDGE_DOCS)
    got = [h.doc_id for h in r.search(QUERY, top_k=2)]
    assert set(got) == {"a", "b"}, f"chain should bridge to b, got {got}"


def test_chain_keeps_the_reranked_head_at_rank_one():
    r = build("chain")
    r.index(BRIDGE_DOCS)
    assert r.search(QUERY, top_k=3)[0].doc_id == "a"


def test_rerank_mode_returns_full_ranking():
    r = build("rerank")
    r.index(BRIDGE_DOCS)
    assert len(r.search(QUERY, top_k=99)) == 5


# --------------------------------------------------------------- decompose
class ScriptedLLM:
    """Splits into sub-questions, then reports the bridge entity."""

    def __init__(self):
        self.prompts = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "Decompose" in prompt:
            return "Who directed Film X?\nWhat award did Jane Roe win?"
        return "Jane Roe"


def test_decompose_uses_subquestions_and_finds_both_hops():
    llm = ScriptedLLM()
    r = build("decompose", llm=llm)
    r.index(BRIDGE_DOCS)
    got = [h.doc_id for h in r.search(QUERY, top_k=2)]
    assert set(got) == {"a", "b"}
    assert any("Decompose" in p for p in llm.prompts)


def test_decompose_substitutes_the_bridge_answer_into_later_hops():
    llm = ScriptedLLM()
    r = build("decompose", llm=llm)
    r.index(BRIDGE_DOCS)
    r.search(QUERY)
    # The second sub-question must be asked with the first answer appended.
    assert any("Jane Roe" in p and "award" in p for p in llm.prompts[1:])


def test_decompose_survives_a_failing_llm():
    def boom(_prompt):
        raise RuntimeError("model offline")

    r = build("decompose", llm=boom)
    r.index(BRIDGE_DOCS)
    assert r.search(QUERY, top_k=1)[0].doc_id == "a"


def test_unknown_bridge_answer_is_discarded():
    """An UNKNOWN answer must not be appended to the next sub-question."""
    r = build("decompose", llm=lambda p: "UNKNOWN")
    r.index(BRIDGE_DOCS)
    assert r._bridge_answer("Who directed Film X?", [0]) == ""


def test_empty_bridge_answer_is_discarded():
    r = build("decompose", llm=lambda p: "   ")
    r.index(BRIDGE_DOCS)
    assert r._bridge_answer("Who directed Film X?", [0]) == ""


def test_unknown_answer_leaves_later_subquestions_unmodified():
    class Unknown(ScriptedLLM):
        def __call__(self, prompt):
            self.prompts.append(prompt)
            return ("Who directed Film X?\nWhat award did they win?\nWhen?"
                    if "Decompose" in prompt else "UNKNOWN")

    llm = Unknown()
    r = build("decompose", llm=llm)
    r.index(BRIDGE_DOCS)
    r.search(QUERY)
    # Bridge substitution appends the answer after the sub-question text.
    assert not any("win? UNKNOWN" in p or "When? UNKNOWN" in p for p in llm.prompts)


def test_malformed_decomposition_falls_back_to_the_query():
    r = build("decompose", llm=lambda p: "not a question at all")
    r.index(BRIDGE_DOCS)
    assert r.search(QUERY, top_k=1)[0].doc_id == "a"


# ------------------------------------------------------------------- hits
def test_hits_are_rank_ordered_from_one():
    r = build()
    r.index(BRIDGE_DOCS)
    hits = r.search(QUERY, top_k=4)
    assert [h.rank for h in hits] == [1, 2, 3, 4]
    assert all(isinstance(h, Hit) for h in hits)


def test_scores_decrease_with_rank():
    r = build()
    r.index(BRIDGE_DOCS)
    scores = [h.score for h in r.search(QUERY, top_k=5)]
    assert scores == sorted(scores, reverse=True)


def test_top_k_caps_results():
    r = build()
    r.index(BRIDGE_DOCS)
    assert len(r.search(QUERY, top_k=2)) == 2


def test_hits_carry_title_and_text():
    r = build()
    r.index(BRIDGE_DOCS)
    top = r.search(QUERY, top_k=1)[0]
    assert top.title == "Film X"
    assert "directed by Jane Roe" in top.text


def test_no_document_is_returned_twice():
    r = build("chain")
    r.index(BRIDGE_DOCS)
    ids = [h.doc_id for h in r.search(QUERY, top_k=5)]
    assert len(ids) == len(set(ids))


# -------------------------------------------------------------- interleave
def test_interleave_is_round_robin_with_first_list_as_backbone():
    assert _interleave([[1, 2, 3], [4, 5, 6]], 6) == [1, 4, 2, 5, 3, 6]


def test_interleave_deduplicates_across_lists():
    assert _interleave([[1, 2], [1, 3]], 4) == [1, 2, 3]


def test_interleave_respects_the_limit():
    assert _interleave([[1, 2, 3], [4, 5, 6]], 3) == [1, 4, 2]


def test_interleave_handles_ragged_and_empty_lists():
    assert _interleave([[1], [], [2, 3]], 9) == [1, 2, 3]
    assert _interleave([], 5) == []


# ------------------------------------------------- precomputed embeddings
def test_index_accepts_precomputed_embeddings():
    r = build()
    emb = FakeEncoder().encode([d.text for d in BRIDGE_DOCS])
    assert r.index(BRIDGE_DOCS, embeddings=emb) == len(BRIDGE_DOCS)
    assert r.search(QUERY, top_k=1)[0].doc_id == "a"


def test_precomputed_embeddings_must_be_row_aligned():
    r = build()
    emb = FakeEncoder().encode([d.text for d in BRIDGE_DOCS[:2]])
    with pytest.raises(ValueError, match="row-aligned"):
        r.index(BRIDGE_DOCS, embeddings=emb)


def test_precomputed_embeddings_match_freshly_encoded_ranking():
    a, b = build("chain"), build("chain")
    a.index(BRIDGE_DOCS)
    b.index(BRIDGE_DOCS, embeddings=FakeEncoder().encode([d.text for d in BRIDGE_DOCS]))
    assert [h.doc_id for h in a.search(QUERY, 5)] == [h.doc_id for h in b.search(QUERY, 5)]


# ------------------------------------------------- anchored residual scorer
class FakeScorer:
    """Stands in for coremem3's BucketScorer: anchor + bounded residual."""

    def __init__(self, geo_scale=10.0, residual=None):
        self.geo_scale = geo_scale
        self.residual = residual  # callable over columns 1..4, or None

    def score_matrix(self, feats):
        base = self.geo_scale * feats[:, 0]
        return base if self.residual is None else base + self.residual(feats[:, 1:])


def test_scorer_feature_matrix_has_the_trained_layout():
    r = build("rerank")
    r.index(BRIDGE_DOCS)
    cos = r._cosine(QUERY)
    idxs = list(range(len(BRIDGE_DOCS)))
    f = r._scorer_features(QUERY, idxs, cos[idxs], r._utility(QUERY, idxs))
    assert f.shape == (len(BRIDGE_DOCS), 5), "column 0 anchor + 4 residual features"
    assert np.allclose(f[:, 0], cos[idxs]), "column 0 must be the raw cosine anchor"


def test_zero_residual_scorer_reproduces_cosine_order():
    """The zero-init guarantee: an untrained scorer must equal pure cosine."""
    r = build("rerank", scorer=FakeScorer())
    r.index(BRIDGE_DOCS)
    dense = build("dense")
    dense.index(BRIDGE_DOCS)
    assert [h.doc_id for h in r.search(QUERY, 5)] == \
           [h.doc_id for h in dense.search(QUERY, 5)]


def test_scorer_replaces_rrf_when_injected():
    """A residual strong enough to reorder proves the scorer is actually used."""
    flip = FakeScorer(residual=lambda x: -100.0 * x[:, 1])  # penalise entity overlap
    r = build("rerank", scorer=flip)
    r.index(BRIDGE_DOCS)
    plain = build("rerank")
    plain.index(BRIDGE_DOCS)
    assert [h.doc_id for h in r.search(QUERY, 5)] != \
           [h.doc_id for h in plain.search(QUERY, 5)]


def test_scorer_features_are_finite_for_a_single_candidate():
    """z-scoring a one-row pool must not produce NaN."""
    r = build("rerank")
    r.index([BRIDGE_DOCS[0]])
    cos = r._cosine(QUERY)
    f = r._scorer_features(QUERY, [0], cos[[0]], r._utility(QUERY, [0]))
    assert np.all(np.isfinite(f))
