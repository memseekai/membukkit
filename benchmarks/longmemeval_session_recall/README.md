# LongMemEval-S session-level retrieval recall (no LLM in the loop)

This is the retrieval-only metric that [gbrain-evals](https://github.com/garrytan/gbrain-evals/blob/main/docs/benchmarks/2026-05-07-longmemeval-s.md)
and [MemPalace](https://github.com/MemPalace/mempalace/blob/main/benchmarks/BENCHMARKS.md)
publish against: on the **original** `xiaowu0162/longmemeval` `_s` split
(500 questions, ~50 sessions per haystack), a question is a hit when at least one
ground-truth `answer_session_id` is among the top-5 retrieved sessions. No reader,
no judge, no LLM anywhere. Same dataset, same K, same n as their reports.

It is scored through MemBukkit's production retrieval stack, not a side harness:
turn-level ingest, cosine candidate generation, optional cross-encoder rerank
fused with cosine by RRF, then sessions ranked by the first appearance of one of
their facts in the relevance-ordered list.

## Results

All MemBukkit rows: n=500, no LLM in the loop, per-question rankings checked in
under `results/`. gbrain and MemPalace rows are their published numbers; we did
not rerun their systems. "Held-out 420" is explained in the next section.

| System | encoder | reranker | R@5 | R@1 | R@3 | R@10 | all-gold@5 | held-out 420 R@5 |
|---|---|---|---|---|---|---|---|---|
| **MemBukkit, no reranker** (`original_openai_depth50_fullscan_norerank`) | `text-embedding-3-large@1536` (gbrain's exact config) | none (`--rerank-select none`) | **99.80%** (499/500) | 93.80 | 99.40 | 99.80 | 94.80 | 99.76 |
| MemBukkit, stock reranker (`..._stockce`) | same | `cross-encoder/ms-marco-MiniLM-L-6-v2`, no fine-tuning | 99.60% (498/500) | 94.60 | 99.00 | 99.80 | 95.00 | 99.52 |
| MemBukkit, shipped reranker (`original_openai_depth50_fullscan`) | same | `reranker_v2` (our fine-tuned MiniLM) | 99.40% (497/500) | 92.40 | 98.20 | 99.80 | 90.40 | 99.52 |
| MemBukkit, as shipped (`original_openai_production`) | same | `reranker_v2`, 33% scan, top_k=10 | 99.20% (496/500) | 93.20 | 98.40 | 99.40 | 93.20 | 99.05 |
| gbrain-hybrid ([report](https://github.com/garrytan/gbrain-evals/blob/main/docs/benchmarks/2026-05-07-longmemeval-s.md)) | `text-embedding-3-large@1536` + BM25 + RRF | none | 97.60% (488/500) | | | | | |
| gbrain-vector | `text-embedding-3-large@1536` | none | 97.40% | | | | | |
| MemPalace raw (ChromaDB) | | none | 96.6% | | | | | |
| MemBukkit, local encoder (`original_local_depth50_fullscan`) | `biencoder_v1` (our 110M fine-tuned mpnet, laptop, $0 API) | `reranker_v2` | 96.40% (482/500) | 86.20 | 94.60 | 98.60 | 82.80 | 96.43 |
| MemBukkit, local encoder, as shipped (`original_local_production`) | same | `reranker_v2`, 33% scan, top_k=10 | 95.40% (477/500) | 86.40 | 93.80 | 97.00 | 84.00 | 95.71 |

`all-gold@5` (every gold session in the top 5) is the stricter multi-session
number nobody publishes; it is here because it is the one that predicts whether
a reader can answer a multi-session question from the retrieved context.

Per-type breakdown for the no-reranker arm is in its `summary.json`; every arm
carries one.

## Read this before quoting it

- **The lift over gbrain is not the reranker, and it is not our fine-tuning.**
  The top row has none of our trained weights in it: gbrain's own embedding
  model, plain cosine, zero fine-tuned components. It beats gbrain-vector
  (same encoder, 97.4) by 2.4 points and gbrain-hybrid by 2.2. What differs is
  what gets embedded: MemBukkit indexes every conversation turn as its own unit
  and ranks sessions by their best turn, where gbrain chunks pages. On this
  benchmark that is the whole gap. It is also a clean negative result for
  BM25+RRF on conversational data, which gbrain's own report half-says (hybrid
  buys 0.2 over vector).
- **Our reranker does not help here and we say so.** With a strong encoder,
  adding the cross-encoder moves R@5 from 99.8 to 99.4 (fine-tuned) or 99.6
  (stock). This matches every other strong-encoder ablation in this repo
  ([`../PAPER_RESULTS.md`](../PAPER_RESULTS.md), [`../beam_ablation/`](../beam_ablation/)):
  the cross-encoder is insurance for weak encoders and a wash or a small tax on
  strong ones. It stays in the default config because the default encoder is the
  110M local one, where it earns its keep.
- **Disclosure: the shipped `reranker_v2` saw 80 LongMemEval questions in
  training.** Its training recipe uses the first 80 instances of the cleaned
  LongMemEval-S set (gold facts as positives, same-haystack hard negatives), so
  for the rows that use `reranker_v2`, 80 of the 500 evaluated questions overlap
  its training data. The "held-out 420" column scores only the other 420. On
  those 80 questions the fine-tuned reranker scores *lower* than on the held-out
  set (98.75 vs 99.52), so the overlap did not inflate the number, but it is a
  contamination and it is why the headline row and the stock-reranker row exist:
  neither has ever seen a LongMemEval question. The bi-encoder `biencoder_v1`
  is trained on PerLTQA only; LongMemEval is held out from it entirely.
- **Our own encoder is below gbrain.** The zero-API arm swaps in our 110M
  fine-tuned bi-encoder: 96.4, 1.2 behind gbrain-hybrid and a statistical tie
  with MemPalace raw (96.6). A laptop model against a hosted embedding model,
  and the gap is what you would expect. It is published here rather than
  hidden because the claim is not "we have a better embedding model"; it is
  that the layer above the encoder is where the recall comes from, and that
  layer works on top of whichever encoder you can afford.
- **Two harness arms, and what they cost.** `--scan-budget 1.0` disables
  topic-bucket gating so we search the whole store, exactly like gbrain does;
  `--rank-depth 50` derives the session ranking from the top-50 relevance-ordered
  facts instead of the production `top_k=10` cut, because ten facts often
  collapse onto fewer than five distinct sessions and recall@5 would be scored
  against an undersized list. The as-shipped rows (33% scan, top_k=10, no
  harness arms) are in the table too: 99.2 with gbrain's encoder, 95.4 with
  ours. That production cut is what produces the
  [82.0 QA accuracy](../../README.md) with a gpt-4o-mini reader on ~3.2k tokens
  of context; it is tuned for answer accuracy per token, not for this metric.
- **Recall@5 is the easy half of the problem.** Four different MemBukkit
  configurations land between 99.2 and 99.8 on it, and gbrain's own vector-only
  arm is at 97.4. On this split, top-5 recall is mostly the encoder plus
  sensible chunking. What separates memory systems is what happens after
  retrieval: how few tokens you can hand the reader and still answer. That is
  why the headline number in the main README is QA accuracy at a fixed token
  budget, not this one.

## Reproduce

Every command runs on the original HuggingFace split, downloads it on first use,
and resumes from `per_question.jsonl` if interrupted. Embeddings are cached in
`.membukkit_emb_cache/` keyed by sha256(text); a cold run of an OpenAI-encoder
arm costs about $2 in embeddings, warm runs cost nothing and take a few minutes
on an M-series laptop. The local-encoder arms take about an hour (encoding every
haystack turn locally dominates).

```bash
E=openai:text-embedding-3-large@1536
R=benchmarks/longmemeval_session_recall/results

# gbrain's encoder, no reranker, nothing fine-tuned (headline row)
uv run python scripts/session_recall.py --encoder $E --rerank-select none \
    --rank-depth 50 --scan-budget 1.0 --output $R/original_openai_depth50_fullscan_norerank

# gbrain's encoder, stock ms-marco reranker (never saw LongMemEval)
uv run python scripts/session_recall.py --encoder $E \
    --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --rank-depth 50 --scan-budget 1.0 --output $R/original_openai_depth50_fullscan_stockce

# gbrain's encoder, our fine-tuned reranker
uv run python scripts/session_recall.py --encoder $E \
    --rank-depth 50 --scan-budget 1.0 --output $R/original_openai_depth50_fullscan

# gbrain's encoder, as-shipped production cut (33% scan, top_k=10)
uv run python scripts/session_recall.py --encoder $E --output $R/original_openai_production

# our 110M local encoder (auto-downloaded from the Hub), zero API calls
uv run python scripts/session_recall.py --rank-depth 50 --scan-budget 1.0 \
    --output $R/original_local_depth50_fullscan
uv run python scripts/session_recall.py --output $R/original_local_production
```

`summary.json` carries aggregate and per-type numbers; `per_question.jsonl`
carries the ranked session ids and gold ids for every question, so any row above
(including the held-out-420 column) can be re-scored without rerunning anything.
