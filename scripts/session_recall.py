"""Session-level retrieval recall on LongMemEval, apples-to-apples with gbrain-evals.

gbrain-evals (github.com/garrytan/gbrain-evals) reports session-level Recall@5 on
the ORIGINAL ``xiaowu0162/longmemeval`` ``_s`` split: a question counts as a hit
when at least one ground-truth ``answer_session_id`` appears among the top-5
retrieved sessions. No reader, no judge, no LLM anywhere in the loop.

This script computes that exact metric through MemBukkit's production retrieval
stack (MemorySystem: bucket routing -> candidate generation -> cross-encoder
rerank). Retrieved facts carry ``source_session`` backpointers (``ingest:{s_idx}``)
that map to ``haystack_session_ids[s_idx]``; sessions are ranked by the first
occurrence of one of their facts in the relevance-ordered candidate list.

Default arm is verbatim-only (raw turns, no distiller): zero LLM calls, the
closest analog to gbrain's embeddings-over-raw-chat setup. Pass
``--lanes verbatim,atomic --distill-cache <path>`` to score the shipped union
config instead (the atomic lane needs distilled facts; cache misses call the LLM).

Counterfactual arms (script-level only, the library/method is untouched):
    --rank-depth N     derive the session ranking from the top-N reranked facts
                       instead of the production top_k cut. With top_k=10, ten
                       facts often collapse onto fewer than 5 distinct sessions,
                       so recall@5 is scored against an undersized list. This is
                       a harness-fairness arm, not a retrieval change.
    --scan-budget X    override the bucket scan budget (1.0 = full scan, i.e.
                       search the whole store like gbrain does)
    --encoder openai:MODEL[@DIMS]   swap the bi-encoder for an OpenAI embedding
                       model (openai:text-embedding-3-large@1536 is gbrain's
                       exact config). Reranker/routing/harness stay ours.

Embeddings are disk-cached (sqlite, sha256(text)-keyed) so repeat arms only pay
for reranking + clustering.

Usage:
    # gbrain-matched arm: their encoder, our stack, full scan, depth-50 ranking
    uv run python scripts/session_recall.py \
        --encoder openai:text-embedding-3-large@1536 --rank-depth 50 --scan-budget 1.0 \
        --output benchmarks/longmemeval_session_recall/results/original_openai_depth50_fullscan

    # zero-API arm: our 110M fine-tuned local encoder
    uv run python scripts/session_recall.py --rank-depth 50 --scan-budget 1.0 \
        --output benchmarks/longmemeval_session_recall/results/original_local_depth50_fullscan
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

K_LEVELS = (1, 3, 5, 10)


def _make_encoder(spec: str, cache_dir: str):
    """Build the encoder for a spec: openai:MODEL[@DIMS], or a local model name/path."""
    if spec.startswith("openai:"):
        from membukkit.models.openai_encoder import make_openai_encoder

        return make_openai_encoder(spec, cache_dir=cache_dir)
    from membukkit.config import ModelConfig
    from membukkit.models.encoder import Encoder
    from membukkit.models.registry import resolve_encoder_path

    return Encoder(resolve_encoder_path(ModelConfig(encoder=spec)))


def _load_data(dataset: str, max_instances: int | None):
    from huggingface_hub import hf_hub_download

    if dataset == "original":
        path = hf_hub_download(
            repo_id="xiaowu0162/longmemeval",
            filename="longmemeval_s",
            repo_type="dataset",
        )
    else:
        path = hf_hub_download(
            repo_id="xiaowu0162/longmemeval-cleaned",
            filename="longmemeval_s_cleaned.json",
            repo_type="dataset",
        )
    with open(path) as f:
        data = json.load(f)
    return data[:max_instances] if max_instances else data


def _rank_sessions(mem, question: str, cfg, rank_depth: int | None = None) -> list[str]:
    """Relevance-ordered distinct source sessions for one query.

    Uses the same routing flags and lane retrieval as production ``search()``,
    but keeps rerank order (search() re-sorts temporally for presentation,
    which would corrupt a rank-based recall metric).
    """
    from membukkit.retrieval.router import (
        is_recommendation_query,
        is_reasoning_query,
        is_temporal_query,
    )

    is_rec = is_recommendation_query(question)
    is_reason = (not is_rec) and is_reasoning_query(question)
    is_temp = is_temporal_query(question)
    k_eff = cfg.reasoning_top_k if (is_reason and cfg.reasoning_top_k > cfg.top_k) else cfg.top_k
    if rank_depth:
        k_eff = rank_depth

    v_cands, a_cands, _ = mem._retrieve_lanes(question, k_eff, is_reason, is_temp, is_rec, None)
    cands = list(v_cands) + list(a_cands)

    id2src = dict(zip(mem._backend._ids, mem._backend._sources))
    ranked: list[str] = []
    for c in cands:
        src = id2src.get(getattr(c, "id", ""), "")
        if src and src not in ranked:
            ranked.append(src)
    return ranked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("original", "cleaned"), default="original")
    ap.add_argument("--lanes", default="verbatim", help="comma list: verbatim[,atomic]")
    ap.add_argument("--distill-cache", default=None, help="required when lanes include atomic")
    ap.add_argument("--distill-model", default="gpt-4o-mini")
    ap.add_argument("--encoder", default="biencoder_v1", help="local model name/path or openai:MODEL[@DIMS]")
    ap.add_argument("--reranker", default="reranker_v2/model")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--reasoning-top-k", type=int, default=30)
    ap.add_argument("--bucket-k", type=int, default=24)
    ap.add_argument("--cand", type=int, default=50)
    ap.add_argument("--rank-depth", type=int, default=None, help="rank sessions from the top-N reranked facts")
    ap.add_argument("--scan-budget", type=float, default=None, help="override all scan budgets (1.0 = full scan)")
    ap.add_argument("--embed-cache-dir", default=".membukkit_emb_cache")
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--output", required=True, help="output directory")
    args = ap.parse_args()

    lanes = tuple(x.strip() for x in args.lanes.split(",") if x.strip())
    if "atomic" in lanes and not args.distill_cache:
        raise SystemExit("--lanes atomic requires --distill-cache (else every session hits the LLM)")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_q_path = out_dir / "per_question.jsonl"

    done: set[str] = set()
    if per_q_path.exists():
        with open(per_q_path) as f:
            done = {json.loads(line)["question_id"] for line in f if line.strip()}
        print(f"resuming: {len(done)} questions already scored")

    data = _load_data(args.dataset, args.max_instances)
    todo = [item for item in data if item["question_id"] not in done]
    print(f"dataset={args.dataset}  n={len(data)}  todo={len(todo)}  lanes={lanes}")

    from membukkit.config import ModelConfig, PromptConfig, RetrievalConfig
    from membukkit.models.registry import resolve_reranker_path
    from membukkit.models.reranker import UtilityReranker
    from membukkit.pipeline import MemorySystem

    encoder = _make_encoder(args.encoder, args.embed_cache_dir)
    reranker = UtilityReranker.load(resolve_reranker_path(ModelConfig(reranker=args.reranker)))
    prompts = PromptConfig.default()

    distiller = None
    if "atomic" in lanes:
        from membukkit.extraction.distiller import FactDistiller
        from membukkit.llm import resolve_llm

        distiller = FactDistiller(resolve_llm(args.distill_model), cache_path=args.distill_cache)

    t_start = time.time()
    with open(per_q_path, "a") as fout:
        for qi, item in enumerate(todo):
            cfg_kwargs = dict(
                num_buckets=args.bucket_k,
                top_k=args.top_k,
                reasoning_top_k=args.reasoning_top_k,
                candidate_pool=args.cand,
                union=True,
                union_lanes=lanes,
            )
            if args.scan_budget is not None:
                cfg_kwargs.update(
                    scan_budget=args.scan_budget,
                    scan_budget_temporal=args.scan_budget,
                    scan_budget_reason=args.scan_budget,
                )
            cfg = RetrievalConfig(**cfg_kwargs)
            mem = MemorySystem(
                encoder=encoder,
                reranker=reranker,
                llm_fn=lambda p: "",
                retrieval=cfg,
                prompts=prompts,
                distiller=distiller,
            )
            mem.ingest(sessions=item["haystack_sessions"], dates=item.get("haystack_dates", []))

            ranked = _rank_sessions(mem, item["question"], cfg, rank_depth=args.rank_depth)
            sess_ids = item["haystack_session_ids"]
            ranked_sids = [
                sess_ids[int(s.split(":")[1])]
                for s in ranked
                if s.startswith("ingest:") and int(s.split(":")[1]) < len(sess_ids)
            ]
            gold = set(item.get("answer_session_ids") or [])

            row = {
                "question_id": item["question_id"],
                "question_type": item["question_type"],
                "is_abstention": str(item["question_id"]).endswith("_abs"),
                "n_gold": len(gold),
                "ranked_sessions": ranked_sids[: max(K_LEVELS)],
                "gold_sessions": sorted(gold),
            }
            for k in K_LEVELS:
                top = set(ranked_sids[:k])
                row[f"any_gold@{k}"] = bool(top & gold)
                row[f"all_gold@{k}"] = gold.issubset(top) if gold else False
            fout.write(json.dumps(row) + "\n")
            fout.flush()

            if (qi + 1) % 10 == 0 or qi == len(todo) - 1:
                dt = time.time() - t_start
                print(
                    f"[{qi + 1}/{len(todo)}] {dt / (qi + 1):.1f}s/q  "
                    f"eta {(len(todo) - qi - 1) * dt / (qi + 1) / 60:.0f}min",
                    flush=True,
                )

    rows = [json.loads(line) for line in open(per_q_path) if line.strip()]
    scored = {r["question_id"]: r for r in rows}.values()  # last write wins

    def _agg(rs):
        out = {"n": len(rs)}
        for k in K_LEVELS:
            out[f"any_gold@{k}"] = sum(r[f"any_gold@{k}"] for r in rs) / len(rs) if rs else float("nan")
            out[f"all_gold@{k}"] = sum(r[f"all_gold@{k}"] for r in rs) / len(rs) if rs else float("nan")
        return out

    non_abs = [r for r in scored if not r["is_abstention"]]
    by_type = defaultdict(list)
    for r in scored:
        by_type[r["question_type"]].append(r)

    summary = {
        "dataset": args.dataset,
        "lanes": list(lanes),
        "encoder": args.encoder,
        "reranker": args.reranker,
        "config": {
            "top_k": args.top_k,
            "reasoning_top_k": args.reasoning_top_k,
            "bucket_k": args.bucket_k,
            "candidate_pool": args.cand,
            "rank_depth": args.rank_depth,
            "scan_budget": args.scan_budget,
        },
        "all_questions": _agg(list(scored)),
        "excluding_abstention": _agg(non_abs),
        "by_type": {t: _agg(rs) for t, rs in sorted(by_type.items())},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 64}\nSESSION-LEVEL RECALL  ({args.dataset} split, lanes={lanes})\n{'=' * 64}")
    for label, agg in (
        ("all (n=%d)" % summary["all_questions"]["n"], summary["all_questions"]),
        ("excl. abstention (n=%d)" % summary["excluding_abstention"]["n"], summary["excluding_abstention"]),
    ):
        print(f"\n{label}")
        for k in K_LEVELS:
            print(
                f"  any-gold@{k:<2d} {agg[f'any_gold@{k}'] * 100:6.2f}%   "
                f"all-gold@{k:<2d} {agg[f'all_gold@{k}'] * 100:6.2f}%"
            )
    print(f"\n{'type':24s} {'n':>4s} {'any@5':>8s} {'all@5':>8s}")
    for t, agg in summary["by_type"].items():
        print(f"{t:24s} {agg['n']:4d} {agg['any_gold@5'] * 100:7.2f}% {agg['all_gold@5'] * 100:7.2f}%")
    print(f"\nwrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
