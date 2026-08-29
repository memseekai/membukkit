"""Shared-corpus multi-hop retrieval benchmark for MemBukkit.

    python -m benchmarks.multihop.run --dataset musique --mode chain
    python -m benchmarks.multihop.run --dataset 2wiki --mode dense --limit 200

Unlike the HotpotQA distractor benchmark, every question here is scored against
the *whole* corpus, so a retriever cannot coast on a pre-filtered candidate set.
The corpus is embedded once and reused across questions.

Scoring is document level: passages carry titles, gold labels are titles, so a
ranked passage list is collapsed to unique titles by first occurrence before any
metric is computed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Dict, List

from benchmarks.common import metrics
from benchmarks.multihop import dataset as ds

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
KS = (1, 2, 5, 10, 20)


def collapse_to_titles(doc_ids: List[str]) -> List[str]:
    out, seen = [], set()
    for d in doc_ids:
        t = ds.title_of(d)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def build_retriever(mode: str, llm_model: str = ""):
    from membukkit.retrieval.rag import RagRetriever

    llm = None
    if mode == "decompose":
        if not llm_model:
            raise SystemExit("--mode decompose needs --llm (e.g. ollama:qwen3:1.7b)")
        llm = _make_llm(llm_model)
    return RagRetriever(mode=mode, llm=llm)


def _make_llm(spec: str):
    """`ollama:<tag>` for a local model, anything else is an OpenAI model id."""
    if spec.startswith("ollama:"):
        import urllib.request

        tag = spec.split("ollama:", 1)[1]

        def call(prompt: str) -> str:
            body = json.dumps({
                "model": tag, "prompt": prompt, "stream": False, "think": False,
                "options": {"temperature": 0, "num_predict": 160},
            }).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/generate", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r).get("response", "") or ""

        return call

    from openai import OpenAI

    client = OpenAI()

    def call(prompt: str) -> str:
        r = client.chat.completions.create(
            model=spec, temperature=0, max_tokens=160,
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""

    return call


def run(dataset: str = "musique", mode: str = "chain", limit: int | None = None,
        seed: int = 42, top_k: int = 20, llm_model: str = "",
        progress_every: int = 25) -> Dict:
    from membukkit.retrieval.rag import Document

    split = ds.load_split(dataset, limit=limit, seed=seed)
    print(f"{split.name}: {len(split.questions)} questions over "
          f"{len(split.passages)} passages, mode={mode}", flush=True)

    r = build_retriever(mode, llm_model)
    t_index = time.perf_counter()
    r.index([Document(doc_id=p.doc_id, text=p.text, title=p.title) for p in split.passages])
    index_s = time.perf_counter() - t_index
    print(f"indexed in {index_s:.0f}s", flush=True)

    per_query: List[Dict] = []
    t0 = time.perf_counter()
    for i, q in enumerate(split.questions, start=1):
        t = time.perf_counter()
        hits = r.search(q.question, top_k=top_k)
        latency_ms = (time.perf_counter() - t) * 1000
        ranked = collapse_to_titles([h.doc_id for h in hits])
        gold = q.gold_titles
        per_query.append({
            "qid": q.qid, "query": q.question, "hop_type": q.hop_type,
            "n_gold": len(gold), "expected_titles": gold,
            "retrieved_titles": ranked[:20],
            "first_relevant_rank": metrics.first_relevant_rank(ranked, gold),
            "latency_ms": latency_ms,
            "metrics": {
                **{f"recall@{k}": metrics.recall_at_k(ranked, gold, k) for k in KS},
                **{f"any@{k}": metrics.any_support_at_k(ranked, gold, k) for k in KS},
                **{f"all@{k}": metrics.all_support_at_k(ranked, gold, k) for k in KS},
                "mrr": metrics.reciprocal_rank(ranked, gold),
                "ndcg@10": metrics.ndcg_at_k(ranked, gold, 10),
            },
        })
        if progress_every and i % progress_every == 0:
            el = time.perf_counter() - t0
            print(f"  {i}/{len(split.questions)} ({el:.0f}s, {el/i:.2f}s/question)", flush=True)

    subsets = {"overall": metrics.aggregate(per_query, ks=KS)}
    for hop in sorted({p["hop_type"] for p in per_query}):
        rows = [p for p in per_query if p["hop_type"] == hop]
        if rows:
            subsets[hop] = metrics.aggregate(rows, ks=KS)

    return {
        "benchmark": f"multihop-{split.name}",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": "membukkit",
        "mode": mode,
        "llm": llm_model or None,
        "dataset": {
            "name": split.name, "source": "official HippoRAG release split",
            "n_questions": len(split.questions), "n_passages": len(split.passages),
            "limit": limit, "seed": seed,
            "qa_sha256": split.qa_sha256, "corpus_sha256": split.corpus_sha256,
            "note": "Shared-corpus retrieval: every question is scored against "
                    "the full corpus, not a per-question candidate set.",
        },
        "index_seconds": index_s,
        "wall_clock_s": time.perf_counter() - t0,
        "subsets": subsets,
        "per_query": per_query,
    }


def print_table(report: Dict) -> None:
    d, subs = report["dataset"], report["subsets"]
    print(f"\n{d['name']} ({d['n_questions']} questions / {d['n_passages']} passages), "
          f"mode={report['mode']}\n")
    hdr = (f"{'subset':<12}{'N':>6}{'R@2':>8}{'R@5':>8}{'All@5':>8}"
           f"{'All@10':>9}{'MRR':>8}{'nDCG@10':>9}{'lat':>9}")
    print(hdr)
    print("-" * len(hdr))
    for name, s in subs.items():
        print(f"{name:<12}{int(s.get('n', 0)):>6}{s.get('recall@2', 0):>8.3f}"
              f"{s.get('recall@5', 0):>8.3f}{s.get('all@5', 0):>8.3f}"
              f"{s.get('all@10', 0):>9.3f}{s.get('mrr', 0):>8.3f}"
              f"{s.get('ndcg@10', 0):>9.3f}{s.get('latency_ms_mean', 0):>7.0f}ms")
    print(f"\nindex {report['index_seconds']:.0f}s | wall {report['wall_clock_s']:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="musique", choices=sorted(ds.ALIASES))
    ap.add_argument("--mode", default="chain",
                    choices=["dense", "rerank", "chain", "decompose"])
    ap.add_argument("--limit", type=int, default=0, help="0 = the full split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--llm", default="", help="decompose only: ollama:<tag> or an OpenAI model id")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    report = run(dataset=args.dataset, mode=args.mode,
                 limit=args.limit or None, seed=args.seed,
                 top_k=args.top_k, llm_model=args.llm)
    print_table(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = args.out or RESULTS_DIR / f"{report['benchmark']}_{args.mode}_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
