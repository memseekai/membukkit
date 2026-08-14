"""Bucket-gated retrieval — topic partition, routing, and RRF.

Core of method v2: instead of scanning the whole bank with a global cosine
prefilter, topic buckets drive candidate generation:

  1. assign : every fact -> KMeans cluster (topic bucket)
  2. route  : query -> bucket scores via cosine(query, centroid)
  3. open buckets in descending score, accumulating facts until >= budget
     fraction of the bank is gathered (whole buckets -> interpretable scan).
  4. cross-encoder reranks only those candidates -> top-k.

Multi-axis routing adds entity and time axes on top of topic buckets.
"""

from __future__ import annotations

import re as _re
from datetime import datetime as _dt
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from membukkit.retrieval.bucket_index import extract_entities, time_bucket_key

SCAN_STATS: List[float] = []


def _build_prototypes(
    fact_embs, by_bucket: Dict[int, List[int]], k_eff: int, k_proto: int, seed: int = 0
):
    """k_proto L2-normalized sub-centroids per bucket (index-time, O(K*k_proto)).

    A single centroid mis-ranks a bucket whose gold fact sits far from the cluster
    mean (heterogeneous clusters); ranking buckets by MAX cos(query, sub-centroid)
    captures that within-bucket spread while routing stays sublinear and the coarse
    K buckets stay interpretable. Returns (proto_mat [P,d] normed, proto_bucket [P])."""
    protos, pbucket = [], []
    for b in range(k_eff):
        ids = by_bucket.get(b, [])
        if not ids:
            continue
        members = fact_embs[ids]
        if len(members) <= k_proto:
            centers = members
        else:
            try:
                from sklearn.cluster import KMeans

                centers = (
                    KMeans(n_clusters=k_proto, random_state=seed, n_init=2)
                    .fit(members)
                    .cluster_centers_
                )
            except Exception:
                centers = members.mean(0, keepdims=True)
        for c in centers:
            protos.append(c)
            pbucket.append(b)
    if not protos:
        return None, None
    P = np.asarray(protos, dtype=np.float32)
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
    return P, np.asarray(pbucket, dtype=np.int64)


def build_topic_partition(fact_embs, k: int = 24, seed: int = 0, k_proto: int = 0) -> Dict:
    """Partition a bank into K KMeans "topic" buckets (cacheable per memory).

    Returns a partition dict {labels, centroids_norm, by_bucket, k_eff, n} (+ proto_mat
    / proto_bucket when k_proto>1, enabling multi-prototype routing). This is the
    expensive step (KMeans); routing a query against it is cheap, so callers with a
    stable bank (e.g. the demo personas) should build ONCE and reuse across queries.
    Uses MiniBatchKMeans for large banks to keep the one-time build fast.
    """
    fact_embs = np.asarray(fact_embs, dtype=np.float32)
    n = len(fact_embs)
    if n == 0:
        return {
            "labels": np.zeros(0, int),
            "centroids_norm": np.zeros((0, 0), np.float32),
            "by_bucket": {},
            "k_eff": 0,
            "n": 0,
            "proto_mat": None,
            "proto_bucket": None,
        }

    k_eff = max(2, min(k, n // 3)) if n >= 6 else 1
    if k_eff <= 1:
        by_bucket = {0: list(range(n))}
        cents = fact_embs.mean(0, keepdims=True)
        cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-8
        return {
            "labels": np.zeros(n, int),
            "centroids_norm": cents,
            "by_bucket": by_bucket,
            "k_eff": 1,
            "n": n,
            "proto_mat": None,
            "proto_bucket": None,
        }

    if n > 5000:
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(
            n_clusters=k_eff, random_state=seed, n_init=3, batch_size=2048, max_iter=100
        )
        labels = km.fit_predict(fact_embs)
    else:
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=k_eff, random_state=seed, n_init=4).fit_predict(fact_embs)

    cents = np.zeros((k_eff, fact_embs.shape[1]), dtype=np.float32)
    by_bucket: Dict[int, List[int]] = {}
    for i, b in enumerate(labels):
        by_bucket.setdefault(int(b), []).append(i)
    for b, ids in by_bucket.items():
        cents[b] = fact_embs[ids].mean(0)
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-8

    proto_mat = proto_bucket = None
    if k_proto and k_proto > 1:
        proto_mat, proto_bucket = _build_prototypes(fact_embs, by_bucket, k_eff, k_proto, seed)
    return {
        "labels": labels,
        "centroids_norm": cents,
        "by_bucket": by_bucket,
        "k_eff": int(k_eff),
        "n": n,
        "proto_mat": proto_mat,
        "proto_bucket": proto_bucket,
    }


def route_topic(
    partition: Dict,
    query_emb,
    budget: float = 0.3,
    record: bool = True,
    exclude: Optional[Sequence[int]] = None,
) -> Tuple[List[int], Dict]:
    """Route a query against a prebuilt topic partition (cheap).

    Rank buckets by cosine(query, centroid) (softmax = inspectable probability per
    bucket), open buckets in descending route prob until >= `budget` fraction of the
    bank is gathered. Those facts are the candidate set (C1 reranks within).

    `exclude` closes buckets entirely (the control interface: topic-scoped
    exclusion / interventions). Excluded buckets can never be opened; the scan
    budget is computed over the remaining bank, and there is NO fallback that
    leaks excluded facts — if every reachable bucket is excluded the candidate
    set is empty (the reader then abstains).
    """
    n = partition.get("n", 0)
    if n == 0:
        return [], {"buckets": [], "scan_frac": 0.0, "n_facts": 0, "n_scanned": 0, "k_total": 0}
    by_bucket = partition["by_bucket"]
    cents = partition["centroids_norm"]
    excl = {int(b) for b in exclude} if exclude else set()
    if partition["k_eff"] <= 1:
        if 0 in excl:
            return [], {
                "buckets": [],
                "scan_frac": 0.0,
                "n_facts": n,
                "n_scanned": 0,
                "k_total": 1,
                "excluded_buckets": sorted(excl),
            }
        if record:
            SCAN_STATS.append(1.0)
        return list(range(n)), {
            "buckets": [{"bucket": 0, "route_prob": 1.0, "size": n}],
            "scan_frac": 1.0,
            "n_facts": n,
            "n_scanned": n,
            "k_total": 1,
        }

    qn = np.asarray(query_emb, dtype=np.float32).ravel()
    qn = qn / (np.linalg.norm(qn) + 1e-8)
    proto_mat = partition.get("proto_mat")
    if proto_mat is not None:
        psims = proto_mat @ qn
        sims = np.full(partition["k_eff"], -np.inf, dtype=np.float32)
        np.maximum.at(sims, partition["proto_bucket"], psims)
        sims[~np.isfinite(sims)] = -1.0
    else:
        sims = cents @ qn
    order = list(np.argsort(sims)[::-1])
    ex = np.exp((sims - sims.max()) / 0.1)
    probs = ex / ex.sum()

    n_excluded = sum(len(by_bucket.get(b, [])) for b in excl)
    n_avail = n - n_excluded
    if n_avail <= 0:
        return [], {
            "buckets": [],
            "scan_frac": 0.0,
            "n_facts": n,
            "n_scanned": 0,
            "k_total": int(partition["k_eff"]),
            "excluded_buckets": sorted(excl),
        }

    target = max(1, int(np.ceil(budget * n_avail)))
    cand_idx: List[int] = []
    opened: List[Dict] = []
    for b in order:
        if int(b) in excl:
            continue
        ids = by_bucket.get(int(b), [])
        if not ids:
            continue
        cand_idx.extend(ids)
        opened.append(
            {
                "bucket": int(b),
                "route_prob": float(probs[b]),
                "cos": float(sims[b]),
                "size": len(ids),
            }
        )
        if len(cand_idx) >= target:
            break
    if not cand_idx and not excl:
        cand_idx = list(range(n))

    scan_frac = len(cand_idx) / n
    if record:
        SCAN_STATS.append(scan_frac)
    trace = {
        "buckets": opened,
        "scan_frac": scan_frac,
        "n_facts": n,
        "n_scanned": len(cand_idx),
        "k_total": int(partition["k_eff"]),
    }
    if excl:
        trace["excluded_buckets"] = sorted(excl)
        trace["n_excluded"] = n_excluded
    return cand_idx, trace


def assign_nearest(centroids_norm, vecs) -> np.ndarray:
    """Assign each (normalized) vector to its nearest centroid. O(n*k), no refit.

    Used at ingest (label new facts) and during a full re-cluster (re-label the
    whole bank), so the expensive KMeans never runs on the write path.
    """
    cents = np.asarray(centroids_norm, dtype=np.float32)
    v = np.asarray(vecs, dtype=np.float32)
    if v.ndim == 1:
        v = v[None, :]
    if cents.size == 0:
        return np.zeros(len(v), dtype=np.int64)
    sims = v @ cents.T  # (n, k)
    return np.argmax(sims, axis=1).astype(np.int64)


def rank_buckets(
    centroids_norm,
    bucket_sizes: Dict[int, int],
    query_emb,
    budget: float,
    n: int,
    exclude: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[Dict], float]:
    """Rank topic buckets for a query from centroids + sizes alone (no member lists).

    Backend-agnostic core of `route_topic`: ranks buckets by cosine(query, centroid)
    with a softmax (the inspectable route probability), opens them in descending
    order until >= `budget` fraction of the bank is covered, and returns the opened
    bucket ids, the trace entries, and the scan fraction. The DB then materialises
    the candidates via a `topic_bucket IN (opened)` filter. `exclude` closes
    buckets entirely (topic-scoped exclusion); the budget is computed over the
    remaining bank.
    """
    cents = np.asarray(centroids_norm, dtype=np.float32)
    k_eff = cents.shape[0]
    if k_eff == 0 or n == 0:
        return [], [], 0.0
    qn = np.asarray(query_emb, dtype=np.float32).ravel()
    qn = qn / (np.linalg.norm(qn) + 1e-8)
    sims = cents @ qn
    order = list(np.argsort(sims)[::-1])
    ex = np.exp((sims - sims.max()) / 0.1)
    probs = ex / ex.sum()

    excl = {int(b) for b in exclude} if exclude else set()
    n_avail = n - sum(int(bucket_sizes.get(b, 0)) for b in excl)
    if n_avail <= 0:
        return [], [], 0.0

    target = max(1, int(np.ceil(budget * n_avail)))
    opened_ids: List[int] = []
    opened: List[Dict] = []
    covered = 0
    for b in order:
        if int(b) in excl:
            continue
        size = int(bucket_sizes.get(int(b), 0))
        if size == 0:
            continue
        opened_ids.append(int(b))
        opened.append(
            {
                "bucket": int(b),
                "route_prob": float(probs[b]),
                "cos": float(sims[b]),
                "size": size,
            }
        )
        covered += size
        if covered >= target:
            break
    scan_frac = covered / n if n else 0.0
    return opened_ids, opened, scan_frac


def rrf_order(util_scores, cos_scores, *extra_scores, k_rrf: int = 60):
    """Reciprocal-rank fusion of cross-encoder (util) and cosine rankings over the SAME
    candidate pool. Returns indices (into the pool) best-first. A fact ranks high if
    EITHER signal ranks it high: cosine catches semantic matches, the cross-encoder catches
    relevance cosine under-ranks, and the bucket gate has already removed off-topic
    distractors. This is the SHIPPED within-region ranker ('hybrid'). k_rrf=60 is standard.

    `extra_scores` takes further same-length ranking signals (the optional BM25
    lexical lane passes one). With none supplied this is the two-signal fusion
    every published number was measured with, unchanged.
    """
    util = np.asarray(util_scores, dtype=np.float64)
    cos = np.asarray(cos_scores, dtype=np.float64)
    n = len(util)

    def _ranks(s):
        r = np.empty(n, dtype=np.int64)
        r[np.argsort(s)[::-1]] = np.arange(n)
        return r

    rrf = 1.0 / (k_rrf + _ranks(util)) + 1.0 / (k_rrf + _ranks(cos))
    for extra in extra_scores:
        rrf = rrf + 1.0 / (k_rrf + _ranks(np.asarray(extra, dtype=np.float64)))
    return list(np.argsort(rrf)[::-1])


def topic_candidates(
    fact_embs,
    query_emb,
    budget: float = 0.3,
    k: int = 24,
    seed: int = 0,
    record: bool = True,
    k_proto: int = 0,
    rerank_cap: int = 0,
) -> Tuple[List[int], Dict]:
    """One-shot topic-bucket gating (build partition + route). Convenient for the
    eval, where each query is a fresh bank. For a stable, reused bank prefer
    build_topic_partition() once + route_topic() per query. k_proto>1 enables
    multi-prototype routing (max-sim over sub-centroids; higher recall, no extra scan).

    rerank_cap>0 caps the opened region to its cosine-top-N BEFORE the caller's C1
    rerank (region/scan-fraction unchanged). This precision step is what lifts v2 to
    0.810 @ ~33%% scan on LongMemEval-S (beating v1's 0.798 @ 100%%; temporal 0.61->0.76):
    buckets pick the scan region, then a v1-style cosine prefilter keeps the cross-encoder
    pool small and high-precision. rerank_cap=0 = no cap (rerank the whole region).
    """
    partition = build_topic_partition(fact_embs, k=k, seed=seed, k_proto=k_proto)
    cand_idx, trace = route_topic(partition, query_emb, budget=budget, record=record)
    if rerank_cap and len(cand_idx) > rerank_cap:
        qn = np.asarray(query_emb, dtype=np.float32).ravel()
        cos = np.asarray(fact_embs)[cand_idx] @ qn
        cand_idx = [cand_idx[j] for j in np.argsort(cos)[::-1][:rerank_cap]]
        trace = {**trace, "n_rerank": len(cand_idx)}
    return cand_idx, trace


_QYEAR_RE = _re.compile(r"\b(19|20)\d{2}\b")
_QMONTH_RE = _re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"(uary|ruary|ch|il|e|y|ust|tember|ober|ember)?\b",
    _re.IGNORECASE,
)
_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def build_multiaxis_partition(texts, times, fact_embs, k: int = 24, seed: int = 0) -> Dict:
    """Topic (KMeans) + entity (symbolic) + time (year-month) buckets, cacheable."""
    topic = build_topic_partition(fact_embs, k=k, seed=seed)
    n = topic["n"]
    entity_to_facts: Dict[str, List[int]] = {}
    time_to_facts: Dict[str, List[int]] = {}
    for i in range(n):
        for e in extract_entities(texts[i]):
            entity_to_facts.setdefault(e, []).append(i)
        tk = time_bucket_key(times[i]) if times is not None and i < len(times) else "unknown"
        time_to_facts.setdefault(tk, []).append(i)
    return {
        "topic": topic,
        "entity_to_facts": entity_to_facts,
        "time_to_facts": time_to_facts,
        "n": n,
    }


def _query_time_keys(query_text: str, time_to_facts: Dict[str, List[int]]) -> List[str]:
    """Map explicit dates in the query to existing year-month bucket keys."""
    years = [m.group(0) for m in _QYEAR_RE.finditer(query_text or "")]
    months = [_MONTHS[m.group(1).lower()[:3]] for m in _QMONTH_RE.finditer(query_text or "")]
    keys: List[str] = []
    present = set(time_to_facts)
    if years and months:
        for y in years:
            for mo in months:
                keys.append(f"{int(y):04d}-{mo:02d}")
    elif years:
        for y in years:
            keys += [k for k in present if k.startswith(f"{int(y):04d}-")]
    elif months:
        for mo in months:
            keys += [k for k in present if k.endswith(f"-{mo:02d}")]
    return [k for k in dict.fromkeys(keys) if k in present]


def route_multiaxis(
    partition: Dict,
    query_text: str,
    query_emb,
    fact_embs,
    budget: float = 0.3,
    temporal: bool = False,
    rerank_cap: int = 200,
    record: bool = True,
) -> Tuple[List[int], Dict]:
    """Union candidates from topic (geometric, budgeted) + entity (exact match) +
    time (explicit-date match) axes, then return the set to hand the cross-encoder.

    The cross-encoder is O(candidates), so on very large banks we cap the TOPIC
    contribution to its cosine-top-`rerank_cap` (intra-region ranking) — but the
    entity/time matches are precise and few, so they are ALWAYS kept (they are the
    needles cosine missed; capping them would defeat multi-axis). scan_frac is
    reported over the full OPENED set (the routing/scan cost), not the capped set.
    """
    n = partition.get("n", 0)
    if n == 0:
        return [], {
            "topic_buckets": [],
            "entity_buckets": [],
            "time_buckets": [],
            "scan_frac": 0.0,
            "n_facts": 0,
            "n_scanned": 0,
        }

    topic_idx, ttrace = route_topic(partition["topic"], query_emb, budget=budget, record=False)

    e2f = partition["entity_to_facts"]
    ent_ids: set = set()
    ent_open = []
    for e in extract_entities(query_text or ""):
        ids = e2f.get(e)
        if ids:
            ent_ids.update(ids)
            ent_open.append({"entity": e, "size": len(ids)})

    t2f = partition["time_to_facts"]
    time_ids: set = set()
    time_open = []
    if temporal:
        for tk in _query_time_keys(query_text, t2f):
            ids = t2f.get(tk)
            if ids:
                time_ids.update(ids)
                time_open.append({"time_bucket": tk, "size": len(ids)})

    opened = set(topic_idx) | ent_ids | time_ids
    scan_frac = (len(opened) / n) if n else 0.0
    if record:
        SCAN_STATS.append(scan_frac)

    topic_keep = list(topic_idx)
    if len(topic_keep) > rerank_cap:
        qn = np.asarray(query_emb, dtype=np.float32).ravel()
        qn = qn / (np.linalg.norm(qn) + 1e-8)
        cos = np.asarray(fact_embs)[topic_keep] @ qn
        topic_keep = [topic_keep[j] for j in np.argsort(cos)[::-1][:rerank_cap]]
    cand = sorted(set(topic_keep) | ent_ids | time_ids)
    if not cand:
        cand = list(range(n))

    trace = {
        "topic_buckets": ttrace.get("buckets", []),
        "entity_buckets": ent_open,
        "time_buckets": time_open,
        "scan_frac": scan_frac,
        "n_facts": n,
        "n_scanned": len(opened),
        "n_rerank": len(cand),
        "k_total": partition["topic"].get("k_eff"),
    }
    return cand, trace


def reset_scan_stats() -> None:
    SCAN_STATS.clear()


def mean_scan_frac() -> float:
    return float(np.mean(SCAN_STATS)) if SCAN_STATS else 0.0
