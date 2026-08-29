"""Document retrieval for RAG, including multi-hop.

``MemorySystem.search`` is built for dated conversational facts: it chunks,
supersedes, and presents results in temporal order. That is the wrong shape for
retrieving *documents* to hand a reader, so this module exposes the document
path directly.

    from membukkit.retrieval.rag import Document, RagRetriever

    r = RagRetriever(mode="chain")
    r.index([Document("d1", "..."), Document("d2", "...")])
    for hit in r.search("who directed the film?", top_k=5):
        print(hit.rank, hit.doc_id, hit.score)

Modes, cheapest first:

``dense``
    Bi-encoder cosine. No reranker, no second hop.
``rerank``
    Cross-encoder utility fused with cosine by RRF. Better at rank 1, but on
    multi-hop questions it systematically demotes the second-hop document,
    because that document is not relevant to the *original* query.
``chain`` (default)
    Rerank to pick the single best document, harvest its entities, append them
    to the query, and rank everything else against that expanded query. This is
    the cheapest configuration that handles a bridge hop, and it needs no LLM.
``decompose``
    ``chain`` plus an LLM loop: split the question into single-hop
    sub-questions, answer each as a short bridge entity, substitute that answer
    into the next sub-question, and interleave the rankings. Needs ``llm``.

Why chain beats pooling: a naive multi-hop pass pools candidates across hops
and then rescores the pool against the original query, which undoes the
expansion. Chaining keeps the hop-1 pick fixed and ranks the remainder against
the *expanded* query instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

MODES = ("dense", "rerank", "chain", "decompose")
_DEFAULT_TOP_K = 10


@dataclass
class Document:
    """One indexable unit. ``text`` is embedded and reranked verbatim."""

    doc_id: str
    text: str
    title: str = ""


@dataclass
class Hit:
    doc_id: str
    score: float
    rank: int  # 1-based
    title: str = ""
    text: str = ""


@dataclass
class _Index:
    docs: List[Document] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.docs)


_DECOMPOSE_PROMPT = (
    "Decompose this multi-hop question into 2-{n} simpler single-hop "
    "sub-questions that must each be answered to answer it. Output ONLY the "
    "sub-questions, one per line, no numbering.\nQuestion: {q}"
)

_BRIDGE_PROMPT = (
    "Answer the question using ONLY the passages, with a SHORT entity or "
    "phrase (no sentence). If the passages do not answer it, reply exactly "
    "UNKNOWN.\nPassages:\n{ctx}\nQuestion: {q}\nShort answer:"
)


class RagRetriever:
    """Document retriever with optional multi-hop expansion.

    Parameters
    ----------
    mode:
        One of :data:`MODES`. See the module docstring.
    encoder, reranker:
        Loaded lazily from the shipped model registry when omitted. Injecting
        them is what makes this testable without downloading models.
    expand_m:
        How many top documents contribute entities to the expanded query.
        Measured best at 1: entities from a single confident document are a
        cleaner bridge than entities pooled from several.
    rerank_cap:
        Cross-encode at most this many candidates, chosen by cosine.
    llm:
        ``fn(prompt) -> str``, required by ``decompose`` mode only.
    """

    def __init__(
        self,
        mode: str = "chain",
        *,
        encoder=None,
        reranker=None,
        expand_m: int = 1,
        rerank_cap: int = 50,
        llm: Optional[Callable[[str], str]] = None,
        max_subq: int = 3,
        encoder_path: Optional[str] = None,
        reranker_path: Optional[str] = None,
        scorer=None,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode == "decompose" and llm is None:
            raise ValueError("mode='decompose' requires an llm callable")
        self.mode = mode
        self.expand_m = max(1, int(expand_m))
        self.rerank_cap = int(rerank_cap)
        self.llm = llm
        self.max_subq = max(1, int(max_subq))
        self._encoder = encoder
        self._reranker = reranker
        self._encoder_path = encoder_path
        self._reranker_path = reranker_path
        # Optional anchored residual scorer replacing RRF in _fuse. It scores
        # geo_scale*cos + MLP(other features) with the cosine weight frozen, so
        # it cannot rank worse than cosine by construction.
        self._scorer = scorer
        self._index = _Index()

    # ------------------------------------------------------------- models
    @property
    def encoder(self):
        if self._encoder is None:
            from membukkit.config import ModelConfig
            from membukkit.models.encoder import Encoder
            from membukkit.models.registry import resolve_encoder_path

            path = self._encoder_path or resolve_encoder_path(ModelConfig())
            logger.info("RagRetriever: loading encoder %s", path)
            self._encoder = Encoder(path)
        return self._encoder

    @property
    def reranker(self):
        if self._reranker is None:
            from membukkit.config import ModelConfig
            from membukkit.models.registry import resolve_reranker_path
            from membukkit.models.reranker import UtilityReranker

            path = self._reranker_path or resolve_reranker_path(ModelConfig())
            logger.info("RagRetriever: loading reranker %s", path)
            self._reranker = UtilityReranker.load(path)
        return self._reranker

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = np.asarray(self.encoder.encode(list(texts)), dtype=np.float64)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0.0, 1.0, norms)

    # -------------------------------------------------------------- index
    def index(self, docs: Sequence[Document],
              embeddings: Optional[np.ndarray] = None) -> int:
        """Embed and store ``docs``, replacing any previous index.

        ``embeddings`` accepts a precomputed, row-aligned matrix so a caller
        comparing several modes over one corpus can encode it once. It must be
        normalized the same way :meth:`_encode` normalizes.
        """
        docs = list(docs)
        if not docs:
            self._index = _Index()
            return 0
        seen = set()
        for d in docs:
            if not d.doc_id:
                raise ValueError("every Document needs a non-empty doc_id")
            if d.doc_id in seen:
                raise ValueError(f"duplicate doc_id {d.doc_id!r}")
            seen.add(d.doc_id)
        if embeddings is None:
            embeddings = self._encode([d.text for d in docs])
        else:
            embeddings = np.asarray(embeddings, dtype=np.float64)
            if embeddings.shape[0] != len(docs):
                raise ValueError(
                    f"embeddings has {embeddings.shape[0]} rows but {len(docs)} "
                    f"documents were given; they must be row-aligned")
        self._index = _Index(docs=docs, embeddings=embeddings)
        return len(docs)

    # ------------------------------------------------------------- search
    def search(self, query: str, top_k: int = _DEFAULT_TOP_K) -> List[Hit]:
        if not self._index.docs:
            return []
        order = self._rank(query)
        return self._hits(order, top_k)

    def _hits(self, order: Sequence[int], top_k: int) -> List[Hit]:
        out: List[Hit] = []
        n = len(self._index)
        for rank, idx in enumerate(list(order)[:top_k], start=1):
            d = self._index.docs[idx]
            # Rank-derived score: modes fuse heterogeneous scales (cosine,
            # cross-encoder utility, RRF), so only the ordering is meaningful.
            out.append(Hit(doc_id=d.doc_id, score=1.0 - (rank - 1) / max(n, 1),
                           rank=rank, title=d.title, text=d.text))
        return out

    def _cosine(self, query: str) -> np.ndarray:
        return self._index.embeddings @ self._encode([query])[0]

    def _utility(self, query: str, idxs: Optional[Sequence[int]] = None) -> np.ndarray:
        texts = [self._index.docs[i].text for i in (idxs if idxs is not None
                                                    else range(len(self._index)))]
        return np.asarray(self.reranker.score(query, texts), dtype=np.float64).ravel()

    def _fuse(self, query: str, cos: np.ndarray) -> List[int]:
        """Cross-encode the cosine-top ``rerank_cap``, then fuse.

        Default fusion is RRF over (utility, cosine) ranks. With a ``scorer``
        injected, fusion instead uses the anchored residual score, which keeps a
        frozen cosine anchor and lets the learned part add only bounded
        corrections.
        """
        from membukkit.retrieval.buckets import rrf_order

        cand = [int(i) for i in np.argsort(cos)[::-1][: self.rerank_cap or len(cos)]]
        util = self._utility(query, cand)
        if self._scorer is not None:
            feats = self._scorer_features(query, cand, cos[cand], util)
            order = np.argsort(self._scorer.score_matrix(feats))[::-1]
            ranked = [cand[int(j)] for j in order]
        else:
            ranked = [cand[j] for j in rrf_order(util, cos[cand])]
        # Anything beyond the rerank cap keeps its cosine order behind the fused head.
        seen = set(ranked)
        ranked += [int(i) for i in np.argsort(cos)[::-1] if int(i) not in seen]
        return ranked

    def _scorer_features(self, query: str, idxs: Sequence[int],
                         cos: np.ndarray, util: np.ndarray) -> np.ndarray:
        """[n, 5] matching the trained layout; column 0 is the frozen anchor."""
        import re

        from membukkit.retrieval.bucket_index import extract_entities

        tok = re.compile(r"[a-z0-9']+")

        def z(v):
            v = np.asarray(v, dtype=np.float32)
            return (v - v.mean()) / (v.std() + 1e-6)

        texts = [self._index.docs[i].text for i in idxs]
        q_ents = extract_entities(query)
        q_toks = set(tok.findall(query.lower()))
        ent = np.array([(len(q_ents & extract_entities(t)) / len(q_ents))
                        if q_ents else 0.0 for t in texts], dtype=np.float32)
        lens = np.array([len(tok.findall(t.lower())) for t in texts], dtype=np.float32)
        lex = np.array([(len(q_toks & set(tok.findall(t.lower()))) / len(q_toks))
                        if q_toks else 0.0 for t in texts], dtype=np.float32)
        return np.stack([np.asarray(cos, dtype=np.float32), z(util), ent,
                         z(np.log1p(lens)), z(lex)], axis=1)

    def _entities(self, idxs: Sequence[int]) -> str:
        from membukkit.retrieval.bucket_index import extract_entities

        ents: set = set()
        for i in idxs:
            ents |= extract_entities(self._index.docs[i].text)
        return " ".join(sorted(ents))

    def _rank(self, query: str) -> List[int]:
        cos = self._cosine(query)
        if self.mode == "dense":
            return [int(i) for i in np.argsort(cos)[::-1]]
        if self.mode == "rerank":
            return self._fuse(query, cos)

        chain = self._chain(query, cos)
        if self.mode == "chain":
            return chain
        return _interleave([chain, self._decompose(query)], len(self._index))

    def _chain(self, query: str, cos: np.ndarray) -> List[int]:
        """Fix the reranked best document, then rank the rest by expanded query."""
        head_order = self._fuse(query, cos)
        heads = head_order[: self.expand_m]
        expanded = f"{query} {self._entities(heads)}".strip()
        cos2 = self._cosine(expanded)
        rest = [i for i in self._fuse(expanded, cos2) if i not in set(heads)]
        return list(heads) + rest

    # ---------------------------------------------------------- decompose
    def _sub_questions(self, query: str) -> List[str]:
        try:
            raw = self.llm(_DECOMPOSE_PROMPT.format(n=self.max_subq, q=query)) or ""
        except Exception:  # a flaky LLM must not take the whole query down
            logger.warning("decomposition call failed; falling back to the query")
            return [query]
        subs = [ln.strip(" -\t").strip() for ln in raw.splitlines()]
        subs = [s for s in subs if s.endswith("?")][: self.max_subq]
        out, seen = [], set()
        for s in [query, *subs]:
            if s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out

    def _bridge_answer(self, question: str, idxs: Sequence[int]) -> str:
        ctx = "\n".join(f"- {self._index.docs[i].text}" for i in idxs)[:4000]
        try:
            raw = (self.llm(_BRIDGE_PROMPT.format(ctx=ctx, q=question)) or "").strip()
        except Exception:
            return ""
        first = raw.splitlines()[0].strip(" .\t") if raw else ""
        return "" if not first or first.upper().startswith("UNKNOWN") else first

    def _decompose(self, query: str) -> List[int]:
        """Iterative decomposition: substitute each answer into the next hop."""
        rankings: List[List[int]] = []
        bridge = ""
        for sub in self._sub_questions(query):
            text = f"{sub} {bridge}".strip() if (bridge and sub != query) else sub
            order = self._fuse(text, self._cosine(text))
            rankings.append(order)
            if sub != query:
                answer = self._bridge_answer(text, order[:3])
                if answer:
                    bridge = f"{bridge} {answer}".strip()
        return _interleave(rankings, len(self._index))


def _interleave(rankings: Sequence[Sequence[int]], limit: int) -> List[int]:
    """Round-robin merge, first list acting as the backbone.

    Interleaving rather than score-pooling is deliberate: the rankings come from
    different queries, so their scores are not on a comparable scale and pooling
    lets one sub-question evict another's correct document.
    """
    out: List[int] = []
    seen: set = set()
    for pos in range(max((len(r) for r in rankings), default=0)):
        for r in rankings:
            if pos < len(r) and r[pos] not in seen:
                seen.add(r[pos])
                out.append(r[pos])
                if len(out) >= limit:
                    return out
    return out
