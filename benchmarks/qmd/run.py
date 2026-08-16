"""Run QMD's benchmark fixture against MemBukkit.

    python -m benchmarks.qmd.run

Reproduces QMD's own protocol: their 6 markdown documents, their 10 queries,
their expected files, and their scorer's semantics (see
``benchmarks/common/qmd_compat.py`` for the quirks). Retrieval only, no answer
generation and no LLM judge.

This fixture is 10 queries over 6 documents. It is a sanity check that
retrieval works end to end, not a basis for comparing systems.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Dict, List

from benchmarks.common import harness, metrics, qmd_compat

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixture"
RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
KS = (1, 3, 5, 10)


def load_fixture() -> Dict:
    return json.loads((FIXTURE_DIR / "example.json").read_text())


def load_documents() -> List[harness.Document]:
    """The 6 markdown files, keyed by filename so ids match QMD's expected_files."""
    docs = []
    for path in sorted((FIXTURE_DIR / "eval-docs").glob("*.md")):
        docs.append(
            harness.Document(doc_id=path.name, text=path.read_text(), title=path.name)
        )
    return docs


def run(top_k: int = 10) -> Dict:
    fixture = load_fixture()
    docs = load_documents()

    mem = harness.build_retrieval_system()
    harness.reset(mem)
    n_facts = harness.ingest_documents(mem, docs)
    harness.assert_undated(mem)

    per_query: List[Dict] = []
    for q in fixture["queries"]:
        # Retrieve deep (all chunks) so the collapsed document ranking can
        # actually reach rank 10; scoring then happens at the k cutoffs.
        ranked, latency_ms, raw_hits = harness.search_documents(mem, q["query"])
        expected = q["expected_files"]

        qmd_scores = qmd_compat.score_results(ranked, expected, q["expected_in_top_k"])
        m = {
            **{f"recall@{k}": metrics.recall_at_k(ranked, expected, k) for k in KS},
            **{f"any@{k}": metrics.any_support_at_k(ranked, expected, k) for k in KS},
            **{f"all@{k}": metrics.all_support_at_k(ranked, expected, k) for k in KS},
            "mrr": metrics.reciprocal_rank(ranked, expected),
            "ndcg@10": metrics.ndcg_at_k(ranked, expected, 10),
            **{f"precision@{k}": metrics.precision_at_k(ranked, expected, k) for k in (1, 5, 10)},
        }
        rank = metrics.first_relevant_rank(ranked, expected)
        per_query.append(
            {
                "id": q["id"],
                "query": q["query"],
                "type": q["type"],
                "expected_docs": expected,
                "expected_in_top_k": q["expected_in_top_k"],
                "retrieved_docs": ranked,
                "first_relevant_rank": rank,
                "passed_qmd_expectation": bool(rank and rank <= q["expected_in_top_k"]),
                "latency_ms": latency_ms,
                "n_chunk_hits": len(raw_hits),
                "n_docs_ranked": len(ranked),
                "metrics": m,
                "qmd_scorer": qmd_scores,
            }
        )

    summary = metrics.aggregate(per_query, ks=KS)
    summary["qmd_expectation_pass_rate"] = metrics.mean(
        [1.0 if q["passed_qmd_expectation"] else 0.0 for q in per_query]
    )
    for key in ("precision_at_k", "recall", "recall_at_1", "recall_at_3", "recall_at_5", "f1"):
        summary[f"qmd_{key}"] = metrics.mean([q["qmd_scorer"][key] for q in per_query])

    manifest = json.loads((FIXTURE_DIR / "MANIFEST.json").read_text())
    return {
        "benchmark": "qmd",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fixture": {
            "upstream_repo": manifest["upstream_repo"],
            "upstream_commit": manifest["upstream_commit"],
            "description": fixture.get("description"),
            "version": fixture.get("version"),
            "n_documents": len(docs),
            "n_queries": len(fixture["queries"]),
            "n_indexed_facts": n_facts,
        },
        "config": harness.config_snapshot(mem),
        "top_k": top_k,
        "summary": summary,
        "per_query": per_query,
    }


def print_table(report: Dict) -> None:
    s = report["summary"]
    print(f"\nQMD fixture  ({report['fixture']['n_queries']} queries, "
          f"{report['fixture']['n_documents']} docs, "
          f"{report['fixture']['n_indexed_facts']} indexed chunks)")
    print(f"upstream {report['fixture']['upstream_commit'][:7]}\n")
    print(f"{'System':<12}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'R@10':>7}{'MRR':>7}"
          f"{'nDCG@10':>9}{'Latency':>10}")
    print("-" * 66)
    print(f"{'MemBukkit':<12}{s.get('recall@1', 0):>7.3f}{s.get('recall@3', 0):>7.3f}"
          f"{s.get('recall@5', 0):>7.3f}{s.get('recall@10', 0):>7.3f}"
          f"{s.get('mrr', 0):>7.3f}{s.get('ndcg@10', 0):>9.3f}"
          f"{s.get('latency_ms_mean', 0):>9.0f}ms")

    print("\nQMD scorer semantics (precision denominator = min(k, |expected|)):")
    print(f"  precision_at_k {s['qmd_precision_at_k']:.3f}   recall {s['qmd_recall']:.3f}   "
          f"f1 {s['qmd_f1']:.3f}")
    print(f"  QMD expected_in_top_k met: {s['qmd_expectation_pass_rate']:.0%} of queries")

    print("\nper query:")
    print(f"  {'id':<22}{'type':<14}{'rank':>5}{'want<=':>7}  {'ok':<3}{'latency':>9}")
    for q in report["per_query"]:
        rank = q["first_relevant_rank"]
        print(f"  {q['id']:<22}{q['type']:<14}{(rank if rank else '-'):>5}"
              f"{q['expected_in_top_k']:>7}  {'y' if q['passed_qmd_expectation'] else 'N':<3}"
              f"{q['latency_ms']:>7.0f}ms")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=10, help="results to retrieve per query")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="JSON output path")
    args = ap.parse_args()

    report = run(top_k=args.top_k)
    print_table(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or RESULTS_DIR / f"qmd_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
