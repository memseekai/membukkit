"""Train + evaluate the cross-encoder utility reranker.

Tests whether a learned utility model retrieves better than cosine.

Pipeline:
  1. Load PerLTQA + LongMemEval banks with per-query gold fact labels.
  2. Split banks into train / test (no bank overlap -> no leakage).
  3. Build (query, fact, label) pairs: gold positives + hard cosine negatives
     + random negatives.
  4. Fine-tune the cross-encoder reranker.
  5. Eval on held-out banks: rerank the top-N cosine candidate pool and
     measure recall@k / MRR vs cosine (focus on the cosine-miss tail).

Data setup:
    Requires PerLTQA and/or LongMemEval. See docs/REPRODUCE.md.

Usage:
    membukkit train-reranker --data-dir data/perltqa --output models/reranker_v2
    # or directly:
    python scripts/train_reranker.py --epochs 2 --output-dir models/reranker_v2
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_ENCODER = None


def _encode(encoder, texts: List[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    parts = []
    for s in range(0, len(texts), batch_size):
        parts.append(encoder.encode(texts[s : s + batch_size], show_progress_bar=False))
    return np.vstack(parts)


def _cosine_ranks(q_emb: np.ndarray, fact_embs: np.ndarray):
    import torch
    import torch.nn.functional as F

    q = F.normalize(torch.tensor(q_emb, dtype=torch.float32), dim=-1)
    F_ = F.normalize(torch.tensor(np.asarray(fact_embs), dtype=torch.float32), dim=-1)
    sims = (F_ @ q).numpy()
    order = np.argsort(sims)[::-1]
    return order, sims


class Bank:
    def __init__(self, bank_id, fact_texts, fact_times, queries, gold_sets, embs):
        self.bank_id = bank_id
        self.fact_texts = fact_texts
        self.fact_times = fact_times
        self.queries = queries
        self.gold_sets = gold_sets
        self.embs = embs


_QUERY_EMB_CACHE: Dict[Tuple[str, str], np.ndarray] = {}


def _cosine_ranks_for_query(bank: Bank, query: str):
    key = (bank.bank_id, query)
    if key not in _QUERY_EMB_CACHE:
        _QUERY_EMB_CACHE[key] = _ENCODER.encode(query, show_progress_bar=False)
    q_emb = _QUERY_EMB_CACHE[key]
    return _cosine_ranks(q_emb, bank.embs)


def load_banks(encoder, max_lme: int, lme_offset: int = 0) -> List[Bank]:
    banks: List[Bank] = []
    if max_lme > 0:
        from membukkit.data.longmemeval import load_longmemeval

        lme = load_longmemeval(max_instances=None)
        insts = lme.instances[lme_offset : lme_offset + max_lme]
        for inst in insts:
            facts = inst.get_facts()
            queries = inst.get_queries()
            if not facts or not queries:
                continue
            gold = set(inst.get_gold_fact_indices())
            if not gold:
                continue
            fact_texts = [f.text for f in facts]
            fact_times = [f.timestamp for f in facts]
            embs = _encode(encoder, fact_texts)
            banks.append(
                Bank(
                    f"lme_{inst.question_id}",
                    fact_texts,
                    fact_times,
                    [queries[0].text],
                    [gold],
                    embs,
                )
            )
    return banks


def build_pairs(
    banks: List[Bank], n_hard_neg: int, n_rand_neg: int, max_pos: int = 5
) -> List[Tuple[str, str, float]]:
    rng = random.Random(13)
    pairs: List[Tuple[str, str, float]] = []
    for bank in banks:
        n = len(bank.fact_texts)
        for q, gold in zip(bank.queries, bank.gold_sets):
            gold_list = [g for g in gold if 0 <= g < n][:max_pos]
            if not gold_list:
                continue
            order, _ = _cosine_ranks_for_query(bank, q)
            hard = [i for i in order if i not in gold][:n_hard_neg]
            pool = [i for i in range(n) if i not in gold and i not in set(hard)]
            rand = rng.sample(pool, min(n_rand_neg, len(pool))) if pool else []
            for g in gold_list:
                pairs.append((q, bank.fact_texts[g], 1.0))
            for neg in hard + rand:
                pairs.append((q, bank.fact_texts[neg], 0.0))
    rng.shuffle(pairs)
    return pairs


def _mrr(order: List[int], gold: set) -> float:
    for rank, idx in enumerate(order):
        if idx in gold:
            return 1.0 / (rank + 1)
    return 0.0


def evaluate_banks(reranker, banks: List[Bank], pool_size: int, top_k: int) -> Dict:
    agg = defaultdict(list)
    tail_total = 0
    tail_recovered = 0

    for bank in banks:
        for q, gold in zip(bank.queries, bank.gold_sets):
            n = len(bank.fact_texts)
            gold = {int(g) for g in gold if 0 <= g < n}
            if not gold:
                continue
            order, sims = _cosine_ranks_for_query(bank, q)
            order = [int(i) for i in order.tolist()]
            cosine_top = order[:top_k]
            pool = order[:pool_size]

            cos_hit = any(g in cosine_top for g in gold)
            cos_recall = len(gold & set(cosine_top)) / len(gold)
            cos_mrr = _mrr(order, gold)

            pool_texts = [bank.fact_texts[i] for i in pool]
            util = reranker.score(q, pool_texts)
            rer_order_pool = [pool[j] for j in np.argsort(util)[::-1]]
            rer_top = rer_order_pool[:top_k]
            rer_hit = any(g in rer_top for g in gold)
            rer_recall = len(gold & set(rer_top)) / len(gold)
            rer_mrr = _mrr(rer_order_pool, gold)

            gold_in_pool = gold & set(pool)
            is_tail = (not cos_hit) and bool(gold_in_pool)
            if is_tail:
                tail_total += 1
                if rer_hit:
                    tail_recovered += 1

            agg["cos_hit"].append(cos_hit)
            agg["rer_hit"].append(rer_hit)
            agg["cos_recall"].append(cos_recall)
            agg["rer_recall"].append(rer_recall)
            agg["cos_mrr"].append(cos_mrr)
            agg["rer_mrr"].append(rer_mrr)

    def m(key):
        return float(np.mean(agg[key])) if agg[key] else 0.0

    return {
        "n_queries": len(agg["cos_hit"]),
        "cosine": {"hit@k": m("cos_hit"), "recall@k": m("cos_recall"), "mrr": m("cos_mrr")},
        "reranker": {"hit@k": m("rer_hit"), "recall@k": m("rer_recall"), "mrr": m("rer_mrr")},
        "cosine_miss_tail": {
            "n_tail": tail_total,
            "recovered": tail_recovered,
            "recovery_rate": (tail_recovered / tail_total) if tail_total else 0.0,
        },
    }


def run(args) -> None:
    global _ENCODER
    from sentence_transformers import SentenceTransformer
    from membukkit.models.reranker import UtilityReranker

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading encoder...")
    _ENCODER = SentenceTransformer(args.encoder)

    logger.info("Loading TRAIN banks...")
    train_banks = load_banks(_ENCODER, args.max_lme_train)
    logger.info("Loading TEST banks...")
    test_banks = load_banks(_ENCODER, args.max_lme_test, lme_offset=args.max_lme_train)
    logger.info(f"Train: {len(train_banks)}  Test: {len(test_banks)}")

    logger.info("Building training pairs...")
    pairs = build_pairs(train_banks, args.n_hard_neg, args.n_rand_neg)
    n_pos = sum(1 for _, _, label in pairs if label == 1.0)
    logger.info(f"Pairs: {len(pairs)} ({n_pos} pos / {len(pairs) - n_pos} neg)")

    reranker = UtilityReranker(base_model=args.base_model, max_length=args.max_length)
    reranker.train(
        pairs, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, output_path=None
    )
    reranker.save(str(out_dir / "model"))

    if not args.skip_eval:
        logger.info("Evaluating on held-out banks...")
        summary = evaluate_banks(reranker, test_banks, args.pool_size, args.top_k)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 60}")
        print("RERANKER vs COSINE (held-out banks)")
        print(f"{'=' * 60}")
        for key in ("hit@k", "recall@k", "mrr"):
            print(
                f"{key:14s} cosine={summary['cosine'][key]:.3f}  "
                f"reranker={summary['reranker'][key]:.3f}"
            )
        t = summary["cosine_miss_tail"]
        print(f"\nTail recovery: {t['recovered']}/{t['n_tail']} = {t['recovery_rate']:.3f}")

    logger.info(f"Model saved to {out_dir / 'model'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Train + eval cross-encoder utility reranker")
    p.add_argument("--max-lme-train", type=int, default=80)
    p.add_argument("--max-lme-test", type=int, default=40)
    p.add_argument("--n-hard-neg", type=int, default=8)
    p.add_argument("--n-rand-neg", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--pool-size", type=int, default=100)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--output-dir", default="models/reranker_v2")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()
    run(args)
