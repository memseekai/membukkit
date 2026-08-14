"""Fixed-substrate retrievers for the multi-hop RAG benchmark.

Each retriever exposes the same contract:
    r = SomeRetriever(...)
    r.index(passages_text, passage_titles)
    idxs = r.retrieve(query, top_k)  -> ranked passage indices

In-process retrievers:
  - DenseRetriever   : EMBEDDER cosine top-k.
  - CoreMemRetriever : bi-encoder -> topic-bucket gating -> C1 cross-encoder
                       UtilityReranker -> hybrid RRF. Supports multi-hop,
                       multi-axis, entity-aware ranking, and query decomposition.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SubstrateEncoder:
    """Wraps a SentenceTransformer EMBEDDER. Encodes corpus once, caches matrix."""

    def __init__(
        self,
        model_name: str,
        query_prompt: str = "",
        trust_remote_code: bool = False,
        max_seq_length: int = 0,
        batch_size: int = 64,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_prompt = query_prompt
        self.batch_size = batch_size
        logger.info(f"Loading EMBEDDER: {model_name}")
        self.model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
        if max_seq_length and max_seq_length > 0:
            self.model.max_seq_length = max_seq_length
            logger.info(f"  capped max_seq_length -> {max_seq_length}")
        self._corpus_emb: Optional[np.ndarray] = None
        self._encode_lock = threading.Lock()

    def encode_corpus(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        if self._corpus_emb is not None and len(self._corpus_emb) == len(texts):
            return self._corpus_emb
        self._corpus_emb = np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size or self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        )
        return self._corpus_emb

    def encode_query(self, query: str) -> np.ndarray:
        q = (self.query_prompt + query) if self.query_prompt else query
        with self._encode_lock:
            return np.asarray(
                self.model.encode(q, normalize_embeddings=True, convert_to_numpy=True)
            )


class DenseRetriever:
    name = "dense"

    def __init__(self, encoder: SubstrateEncoder):
        self.enc = encoder
        self._emb: Optional[np.ndarray] = None

    def index(self, passages_text: List[str], passage_titles: List[str]) -> None:
        self._emb = self.enc.encode_corpus(passages_text)

    def retrieve(self, query: str, top_k: int) -> List[int]:
        q = self.enc.encode_query(query)
        scores = self._emb @ q
        k = min(top_k, len(scores))
        return list(np.argsort(scores)[::-1][:k])


class CoreMemRetriever:
    """CoreMem retrieval stack: bi-encoder -> topic-bucket gating -> cross-encoder
    UtilityReranker -> hybrid RRF. Supports multi-hop iterative retrieval,
    multi-axis indexing, entity-aware ranking, and query decomposition."""

    name = "coremem"

    def __init__(
        self,
        encoder_path: str,
        reranker_path: str,
        budget: float = 0.3,
        bucket_k: int = 24,
        k_proto: int = 0,
        rerank_cap: int = 50,
        fusion: str = "rrf",
        hops: int = 1,
        expand_m: int = 3,
        expand_mode: str = "entity",
        axes: str = "topic",
        temporal: bool = False,
        entity_cap: int = 50,
        entity_rank: bool = False,
        entity_min: int = 1,
        decompose: bool = False,
        max_subq: int = 4,
        decompose_fuse: str = "interleave",
        decompose_iter: bool = True,
        decompose_retrieval: str = "full_cosine",
        llm_fn=None,
        query_prompt: str = "",
        trust_remote_code: bool = False,
        max_seq_length: int = 0,
        batch_size: int = 64,
        encoder: Optional["SubstrateEncoder"] = None,
    ):
        self.encoder_path = encoder_path
        self.reranker_path = reranker_path
        self.budget = budget
        self.bucket_k = bucket_k
        self.k_proto = k_proto
        self.rerank_cap = rerank_cap
        self.fusion = fusion
        self.hops = max(1, int(hops))
        self.expand_m = max(1, int(expand_m))
        self.expand_mode = expand_mode
        self.axes = axes
        self.temporal = bool(temporal)
        self.entity_cap = int(entity_cap)
        self.entity_rank = bool(entity_rank)
        self.entity_min = max(1, int(entity_min))
        self.decompose = bool(decompose)
        self.max_subq = max(1, int(max_subq))
        self.decompose_fuse = decompose_fuse
        self.decompose_iter = bool(decompose_iter)
        if decompose_retrieval not in ("full_cosine", "bucket"):
            raise ValueError(
                "decompose_retrieval must be 'full_cosine' (historical path) "
                "or 'bucket' (route every sub-question through the configured selector)"
            )
        if decompose_retrieval == "bucket" and decompose_fuse != "interleave":
            raise ValueError("bucketed decomposition currently requires interleave fusion")
        self.decompose_retrieval = decompose_retrieval
        self.llm_fn = llm_fn
        self._mpart = None
        self.query_prompt = query_prompt
        self.trust_remote_code = trust_remote_code
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self._enc: Optional[SubstrateEncoder] = encoder
        self._reranker = None
        self._fe: Optional[np.ndarray] = None
        self._texts: Optional[List[str]] = None
        self._partition = None
        self._scan_fracs: List[float] = []
        self._rerank_lock = threading.Lock()

    def index(self, passages_text: List[str], passage_titles: List[str]) -> None:
        from membukkit.retrieval.buckets import build_topic_partition

        if self._enc is None:
            self._enc = SubstrateEncoder(
                self.encoder_path,
                query_prompt=self.query_prompt,
                trust_remote_code=self.trust_remote_code,
                max_seq_length=self.max_seq_length,
                batch_size=self.batch_size,
            )
        self._texts = passages_text
        self._fe = self._enc.encode_corpus(passages_text)
        if self.axes == "multi":
            from membukkit.retrieval.buckets import build_multiaxis_partition

            times = self._extract_times(passages_text) if self.temporal else None
            logger.info(
                f"CoreMem: building MULTI-AXIS partition (topic+entity"
                f"{'+time' if self.temporal else ''}, k={self.bucket_k}) over "
                f"{len(passages_text)} passages..."
            )
            self._mpart = build_multiaxis_partition(passages_text, times, self._fe, k=self.bucket_k)
            self._partition = self._mpart["topic"]
        else:
            logger.info(
                f"CoreMem: building topic partition (k={self.bucket_k}) over "
                f"{len(passages_text)} passages..."
            )
            self._partition = build_topic_partition(self._fe, k=self.bucket_k, k_proto=self.k_proto)
        if self.fusion in ("rrf", "rerank"):
            from membukkit.models.reranker import UtilityReranker

            logger.info(f"CoreMem: loading reranker {self.reranker_path} (fusion={self.fusion})")
            self._reranker = UtilityReranker.load(self.reranker_path)
        else:
            logger.info(f"CoreMem: fusion={self.fusion} -> no reranker loaded")

    _YEAR_RE = None

    def _extract_times(self, texts: List[str]):
        import re
        from datetime import datetime

        if CoreMemRetriever._YEAR_RE is None:
            CoreMemRetriever._YEAR_RE = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")
        out = []
        for t in texts:
            m = CoreMemRetriever._YEAR_RE.search(t or "")
            out.append(datetime(int(m.group(0)), 1, 1) if m else None)
        return out

    _DECOMP_PROMPT = (
        "Decompose this multi-hop question into 2-{n} simpler single-hop sub-questions that "
        "must each be answered to answer it. Output ONLY the sub-questions, one per line, no "
        "numbering.\nQuestion: {q}"
    )

    def _decompose(self, query: str) -> List[str]:
        fn = self.llm_fn
        if fn is None:
            return [query]
        try:
            out = fn(self._DECOMP_PROMPT.format(n=self.max_subq, q=query)) or ""
        except Exception:
            return [query]
        subs = [ln.strip(" -\t").strip() for ln in out.splitlines()]
        subs = [s for s in subs if s and s.endswith("?")][: self.max_subq]
        seen, ordered = set(), []
        for s in [query] + subs:
            k = s.lower()
            if k not in seen:
                seen.add(k)
                ordered.append(s)
        return ordered

    _SUBANS_PROMPT = (
        "Answer the question using ONLY the passages, with a SHORT entity or phrase (no "
        "sentence). If the passages do not answer it, reply exactly UNKNOWN.\n"
        "Passages:\n{ctx}\nQuestion: {q}\nShort answer:"
    )

    def _short_answer(self, q: str, passages: List[str]) -> str:
        fn = self.llm_fn
        if fn is None:
            return ""
        ctx = "\n".join(f"- {p}" for p in passages)[:4000]
        try:
            out = (fn(self._SUBANS_PROMPT.format(ctx=ctx, q=q)) or "").strip()
        except Exception:
            return ""
        a = out.splitlines()[0].strip(" .\t") if out else ""
        return "" if (not a or a.upper().startswith("UNKNOWN")) else a

    @staticmethod
    def _interleave(ranked_lists: List[List[int]], top_k: int) -> List[int]:
        out: List[int] = []
        seen: set = set()
        pos = 0
        while len(out) < top_k:
            progressed = False
            for lst in ranked_lists:
                if pos < len(lst):
                    progressed = True
                    ci = lst[pos]
                    if ci not in seen:
                        seen.add(ci)
                        out.append(ci)
                        if len(out) >= top_k:
                            break
            if not progressed:
                break
            pos += 1
        return out

    def _score_reranker(self, query: str, texts: List[str]) -> np.ndarray:
        with self._rerank_lock:
            return np.asarray(self._reranker.score(query, texts))

    def retrieve(self, query: str, top_k: int) -> List[int]:
        assert self._enc is not None and self._fe is not None and self._texts is not None
        if not (self.decompose and self.llm_fn):
            return self._retrieve_core(query, top_k)
        cap = self.rerank_cap or 100
        subs = self._decompose(query)
        ranked_lists: List[List[int]] = []
        bridge = ""
        for sq in subs:
            q_text = sq
            if self.decompose_iter and bridge and sq != query:
                q_text = f"{sq} {bridge}"
            if self.decompose_retrieval == "bucket":
                order = self._retrieve_core(q_text, cap)
            else:
                qe = self._enc.encode_query(q_text)
                cos = self._fe @ qe
                order = [int(i) for i in np.argsort(cos)[::-1][:cap]]
            ranked_lists.append(order)
            if self.decompose_iter and sq != query:
                ans = self._short_answer(q_text, [self._texts[i] for i in order[:3]])
                if ans:
                    bridge = (bridge + " " + ans).strip() if bridge else ans

        if self.decompose_fuse == "maxpool":
            pooled: Dict[int, float] = {}
            for sq, order in zip(subs, ranked_lists):
                qe = self._enc.encode_query(sq)
                cos = self._fe @ qe
                for ci in order:
                    c = float(cos[ci])
                    if c > pooled.get(ci, -1e9):
                        pooled[ci] = c
            return sorted(pooled.keys(), key=lambda i: pooled[i], reverse=True)[:top_k]
        return self._interleave(ranked_lists, top_k)

    def _retrieve_core(self, query: str, top_k: int) -> List[int]:
        assert self._enc is not None and self._fe is not None and self._texts is not None
        assert self._partition is not None
        if self.axes == "multi":
            return self._retrieve_multiaxis(query, top_k)
        if self.hops > 1:
            return self._retrieve_multihop(query, top_k)
        from membukkit.retrieval.buckets import route_topic, rrf_order

        qe = self._enc.encode_query(query)
        cand_idx, trace = route_topic(self._partition, qe, budget=self.budget, record=False)
        self._scan_fracs.append(float(trace.get("scan_frac", 0.0)))
        if not cand_idx:
            cand_idx = list(range(len(self._texts)))
        cos = self._fe[cand_idx] @ qe
        if self.rerank_cap and len(cand_idx) > self.rerank_cap:
            keep = np.argsort(cos)[::-1][: self.rerank_cap]
            cand_idx = [cand_idx[j] for j in keep]
            cos = cos[keep]
        if self.fusion == "cosine":
            order = list(np.argsort(cos)[::-1])
        else:
            assert self._reranker is not None
            util = self._score_reranker(query, [self._texts[i] for i in cand_idx])
            if self.fusion == "rerank":
                order = list(np.argsort(util)[::-1])
            else:
                order = rrf_order(util, cos)
        return [cand_idx[j] for j in order[:top_k]]

    def scan_summary(self) -> Dict[str, float]:
        """Observed opened-region fractions across all routed (sub-)queries."""
        if not self._scan_fracs:
            return {"scan_fraction_mean": 1.0, "scan_fraction_observations": 0}
        vals = np.asarray(self._scan_fracs, dtype=np.float64)
        return {
            "scan_fraction_mean": float(vals.mean()),
            "scan_fraction_min": float(vals.min()),
            "scan_fraction_max": float(vals.max()),
            "scan_fraction_observations": int(len(vals)),
        }

    def _retrieve_multihop(self, query: str, top_k: int) -> List[int]:
        from membukkit.retrieval.bucket_index import extract_entities
        from membukkit.retrieval.buckets import rrf_order

        assert self._enc is not None and self._fe is not None and self._texts is not None
        cap = self.rerank_cap or 100
        pooled: Dict[int, float] = {}
        cur_qe = self._enc.encode_query(query)
        accum_ents: set = set()
        for h in range(self.hops):
            cos = self._fe @ cur_qe
            top = np.argsort(cos)[::-1][:cap]
            for i in top:
                ci = int(i)
                c = float(cos[ci])
                if c > pooled.get(ci, -1e9):
                    pooled[ci] = c
            if h < self.hops - 1:
                top_m = [int(i) for i in np.argsort(cos)[::-1][: self.expand_m]]
                for i in top_m:
                    accum_ents |= extract_entities(self._texts[i])
                parts = [query]
                if self.expand_mode in ("entity", "both") and accum_ents:
                    parts.append(" ".join(sorted(accum_ents)))
                if self.expand_mode in ("passage", "both"):
                    parts += [self._texts[i] for i in top_m]
                cur_qe = self._enc.encode_query(" ".join(parts))

        order = sorted(pooled.keys(), key=lambda i: pooled[i], reverse=True)
        if self.fusion == "cosine" or self._reranker is None:
            return order[:top_k]
        cand_idx = order[:cap]
        cos_p = np.asarray([pooled[i] for i in cand_idx])
        util = self._score_reranker(query, [self._texts[i] for i in cand_idx])
        if self.fusion == "rerank":
            o = list(np.argsort(util)[::-1])
        else:
            o = rrf_order(util, cos_p)
        return [cand_idx[j] for j in o[:top_k]]

    def _retrieve_multiaxis(self, query: str, top_k: int) -> List[int]:
        from membukkit.retrieval.bucket_index import extract_entities
        from membukkit.retrieval.buckets import route_multiaxis, rrf_order

        assert self._enc is not None and self._fe is not None and self._texts is not None
        assert self._mpart is not None
        cap = self.rerank_cap or 100
        qe = self._enc.encode_query(query)
        cand, _ = route_multiaxis(
            self._mpart,
            query,
            qe,
            self._fe,
            budget=self.budget,
            temporal=self.temporal,
            rerank_cap=cap,
            record=False,
        )
        if not cand:
            cand = list(range(len(self._texts)))
        cos = self._fe[cand] @ qe
        pooled: Dict[int, float] = {int(i): float(c) for i, c in zip(cand, cos)}
        ent_match: Dict[int, int] = {}

        if self.hops > 1:
            top_m = [cand[j] for j in np.argsort(cos)[::-1][: self.expand_m]]
            ents: set = set()
            for i in top_m:
                ents |= extract_entities(self._texts[i])
            e2f = self._mpart.get("entity_to_facts", {})
            bridge: set = set()
            for e in ents:
                ids = e2f.get(e)
                if ids and len(ids) <= self.entity_cap:
                    bridge.update(ids)
                    for i in ids:
                        ent_match[i] = ent_match.get(i, 0) + 1
            if ents:
                qe2 = self._enc.encode_query(query + " " + " ".join(sorted(ents)))
                cos2 = self._fe @ qe2
                soft = [int(i) for i in np.argsort(cos2)[::-1][:cap]]
                for i in list(bridge) + soft:
                    c = float(max(self._fe[i] @ qe, cos2[i]))
                    if c > pooled.get(i, -1e9):
                        pooled[i] = c
            else:
                for i in bridge:
                    c = float(self._fe[i] @ qe)
                    if c > pooled.get(i, -1e9):
                        pooled[i] = c

        order = sorted(pooled.keys(), key=lambda i: pooled[i], reverse=True)
        if self.fusion == "cosine" or self._reranker is None:
            if self.entity_rank and self.hops > 1 and ent_match:
                cl = list(pooled.keys())
                cos_arr = np.asarray([pooled[i] for i in cl], dtype=np.float64)
                ent_arr = np.asarray(
                    [
                        ent_match.get(i, 0) if ent_match.get(i, 0) >= self.entity_min else 0
                        for i in cl
                    ],
                    dtype=np.float64,
                )
                o = rrf_order(ent_arr, cos_arr)
                return [cl[j] for j in o[:top_k]]
            return order[:top_k]
        cand_idx = order[:cap]
        cos_p = np.asarray([pooled[i] for i in cand_idx])
        util = self._score_reranker(query, [self._texts[i] for i in cand_idx])
        if self.fusion == "rerank":
            o = list(np.argsort(util)[::-1])
        else:
            o = rrf_order(util, cos_p)
        return [cand_idx[j] for j in o[:top_k]]
