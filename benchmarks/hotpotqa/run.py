"""HotpotQA distractor retrieval benchmark for MemBukkit.

    python -m benchmarks.hotpotqa.run --limit 100 --seed 42
    python -m benchmarks.hotpotqa.run --split validation      # full split

Extends the QMD-style document-retrieval protocol to a multi-document setting.
Each question is scored against its own 10-paragraph candidate set, of which 2
are gold supporting documents, so this measures candidate-set retrieval and not
retrieval over all of Wikipedia.

Scoring is document level. MemBukkit retrieves chunks, so ranked chunks are
collapsed to unique documents by first occurrence before any metric is computed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Dict, List

from benchmarks.common import harness, metrics
from benchmarks.hotpotqa import dataset as ds

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
KS = (1, 3, 5, 10)


def _subset_summary(rows: List[Dict]) -> Dict:
    if not rows:
        return {}
    out = metrics.aggregate(rows, ks=KS)
    out["avg_gold_docs"] = metrics.mean([float(r["n_gold"]) for r in rows])
    return out


def run(
    split: str = "validation",
    limit: int | None = 100,
    seed: int = 42,
    progress_every: int = 25,
) -> Dict:
    questions = ds.load_questions(split=split, limit=limit, seed=seed)
    print(f"loaded {len(questions)} questions from HotpotQA {split} (distractor)", flush=True)

    mem = harness.build_retrieval_system()
    per_query: List[Dict] = []
    t0 = time.perf_counter()

    for i, q in enumerate(questions, start=1):
        # Per-question isolation: only this question's candidate set is indexed,
        # so no other question's paragraphs can leak into its ranking.
        harness.reset(mem)
        docs = [
            harness.Document(doc_id=d["doc_id"], text=d["text"], title=d["title"])
            for d in q.docs
        ]
        harness.ingest_documents(mem, docs)
        if i == 1:
            harness.assert_undated(mem)

        gold = ds.gold_doc_ids(q)
        ranked, latency_ms, raw_hits = harness.search_documents(mem, q.question)

        m = {
            **{f"recall@{k}": metrics.recall_at_k(ranked, gold, k) for k in KS},
            **{f"any@{k}": metrics.any_support_at_k(ranked, gold, k) for k in KS},
            **{f"all@{k}": metrics.all_support_at_k(ranked, gold, k) for k in KS},
            "mrr": metrics.reciprocal_rank(ranked, gold),
            "ndcg@10": metrics.ndcg_at_k(ranked, gold, 10),
        }
        per_query.append(
            {
                "qid": q.qid,
                "query": q.question,
                "type": q.qtype,
                "level": q.level,
                "n_gold": q.n_gold,
                "n_candidates": len(q.docs),
                "expected_docs": gold,
                "expected_titles": q.gold_titles,
                "retrieved_docs": ranked[:10],
                "first_relevant_rank": metrics.first_relevant_rank(ranked, gold),
                "latency_ms": latency_ms,
                "n_chunk_hits": len(raw_hits),
                "n_docs_ranked": len(ranked),
                "metrics": m,
            }
        )
        if progress_every and i % progress_every == 0:
            el = time.perf_counter() - t0
            # flush: stdout is block-buffered when redirected to a file, so
            # without this a long run shows no progress until it finishes.
            print(
                f"  {i}/{len(questions)}  ({el:.0f}s elapsed, {el / i:.2f}s/question)",
                flush=True,
            )

    subsets = {
        "overall": _subset_summary(per_query),
        "bridge": _subset_summary([r for r in per_query if r["type"] == "bridge"]),
        "comparison": _subset_summary([r for r in per_query if r["type"] == "comparison"]),
    }

    return {
        "benchmark": "hotpotqa-distractor",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "name": "HotpotQA",
            "setting": "distractor",
            "split": split,
            "limit": limit,
            "seed": seed,
            "n_questions": len(questions),
            "note": (
                "Candidate-set retrieval: each question is scored against its own "
                "10 paragraphs (2 gold + 8 distractors), not against all of Wikipedia."
            ),
        },
        "config": harness.config_snapshot(mem),
        "wall_clock_s": time.perf_counter() - t0,
        "subsets": subsets,
        "per_query": per_query,
    }


def print_table(report: Dict) -> None:
    d, subs = report["dataset"], report["subsets"]
    o = subs["overall"]
    print(f"\nHotpotQA {d['split']} (distractor), {d['n_questions']} questions, "
          f"seed {d['seed']}")
    print(f"candidate-set retrieval: {report['per_query'][0]['n_candidates']} docs/question\n")

    print(f"{'System':<12}{'Any@1':>8}{'Any@3':>8}{'All@3':>8}{'All@5':>8}"
          f"{'All@10':>8}{'MRR':>8}{'Latency':>10}")
    print("-" * 70)
    print(f"{'MemBukkit':<12}{o.get('any@1', 0):>8.3f}{o.get('any@3', 0):>8.3f}"
          f"{o.get('all@3', 0):>8.3f}{o.get('all@5', 0):>8.3f}{o.get('all@10', 0):>8.3f}"
          f"{o.get('mrr', 0):>8.3f}{o.get('latency_ms_mean', 0):>8.0f}ms")

    print(f"\n{'Subset':<14}{'N':>6}{'Any@5':>9}{'All@5':>9}{'MRR':>9}{'nDCG@10':>10}"
          f"{'avg gold':>10}")
    print("-" * 68)
    for name in ("overall", "bridge", "comparison"):
        s = subs.get(name) or {}
        if not s:
            continue
        print(f"{name:<14}{int(s.get('n', 0)):>6}{s.get('any@5', 0):>9.3f}"
              f"{s.get('all@5', 0):>9.3f}{s.get('mrr', 0):>9.3f}"
              f"{s.get('ndcg@10', 0):>10.3f}{s.get('avg_gold_docs', 0):>10.2f}")

    lat = o.get("latency_ms_median", 0)
    print(f"\nlatency mean {o.get('latency_ms_mean', 0):.0f}ms / median {lat:.0f}ms  "
          f"| wall clock {report['wall_clock_s']:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="validation", choices=["validation", "train"])
    ap.add_argument("--limit", type=int, default=100,
                    help="deterministic sample size; omit --limit 0 for the full split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    limit = None if args.limit in (0, -1) else args.limit
    report = run(split=args.split, limit=limit, seed=args.seed)
    print_table(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = args.out or RESULTS_DIR / f"hotpotqa_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
