"""Document-level retrieval metrics.

Pure functions over ranked document-id lists, so they can be tested without a
retriever. Everything here takes ``ranked`` (unique doc ids, best first, already
collapsed from chunks by :mod:`benchmarks.common.dedup`) and ``gold`` (the set
of relevant doc ids for that query).

Two families, because the two benchmarks ask different questions:

- ``recall_at_k`` is the fraction of gold documents found in the top k. This is
  what QMD's scorer reports, and it is what you want when a query has one
  right answer.
- ``any_support_at_k`` / ``all_support_at_k`` are the HotpotQA framing: did we
  find *at least one* supporting document, and did we find *all* of them. With
  a single gold document the three collapse to the same number; with two or
  more they diverge, which is the whole point of the multi-hop extension.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Set


def _top(ranked: Sequence[str], k: int) -> List[str]:
    return list(ranked[:k]) if k > 0 else []


def recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Fraction of gold documents present in the top k."""
    gold_set: Set[str] = set(gold)
    if not gold_set:
        return 0.0
    return len(gold_set & set(_top(ranked, k))) / len(gold_set)


def any_support_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """1.0 when at least one gold document is in the top k."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return 1.0 if gold_set & set(_top(ranked, k)) else 0.0


def all_support_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """1.0 only when every gold document is in the top k."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return 1.0 if gold_set <= set(_top(ranked, k)) else 0.0


def first_relevant_rank(ranked: Sequence[str], gold: Iterable[str]) -> int | None:
    """1-based rank of the first gold document, or None if absent."""
    gold_set = set(gold)
    for i, doc in enumerate(ranked, start=1):
        if doc in gold_set:
            return i
    return None


def reciprocal_rank(ranked: Sequence[str], gold: Iterable[str]) -> float:
    """1/rank of the first gold document; 0.0 when none was retrieved."""
    rank = first_relevant_rank(ranked, gold)
    return 1.0 / rank if rank else 0.0


def precision_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Textbook precision@k: relevant hits divided by k.

    Note this is *not* what QMD reports; see
    :func:`benchmarks.common.qmd_compat.qmd_precision_at_k`.
    """
    if k <= 0:
        return 0.0
    gold_set = set(gold)
    return len(gold_set & set(_top(ranked, k))) / k


def ndcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Binary-relevance nDCG@k with the standard 1/log2(rank+1) discount."""
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, doc in enumerate(_top(ranked, k), start=1)
        if doc in gold_set
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold_set), k) + 1))
    return dcg / ideal if ideal else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0


def aggregate(per_query: Sequence[Dict], ks: Sequence[int] = (1, 3, 5, 10)) -> Dict:
    """Average the per-query metric dicts produced by the runners."""
    if not per_query:
        return {}
    out: Dict[str, float] = {"n": len(per_query)}
    keys = [f"{fam}@{k}" for k in ks for fam in ("recall", "any", "all")]
    for key in keys:
        vals = [q["metrics"][key] for q in per_query if key in q["metrics"]]
        if vals:
            out[key] = mean(vals)
    for key in ("mrr", "ndcg@10"):
        vals = [q["metrics"][key] for q in per_query if key in q["metrics"]]
        if vals:
            out[key] = mean(vals)
    lat = [q["latency_ms"] for q in per_query if q.get("latency_ms") is not None]
    if lat:
        out["latency_ms_mean"] = mean(lat)
        out["latency_ms_median"] = median(lat)
    return out
