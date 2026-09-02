# LongMemEval-S session-level retrieval recall (no LLM in the loop)

This is the retrieval-only metric that [gbrain-evals](https://github.com/garrytan/gbrain-evals/blob/main/docs/benchmarks/2026-05-07-longmemeval-s.md)
and [MemPalace](https://github.com/MemPalace/mempalace/blob/main/benchmarks/BENCHMARKS.md)
publish against: on the **original** `xiaowu0162/longmemeval` `_s` split
(500 questions, ~50 sessions per haystack), a question is a hit when at least one
ground-truth `answer_session_id` is among the top-5 retrieved sessions. No reader,
no judge, no LLM anywhere. Same dataset, same K, same n as their reports.

It is scored through MemBukkit's production retrieval stack, not a side harness:
turn-level ingest, cosine candidate generation, cross-encoder rerank fused with
cosine by RRF (`--rerank-select hybrid`), then sessions ranked by the first
appearance of one of their facts in the reranked list.

## Results

| System | encoder | R@5 | R@1 | R@3 | R@10 | LLM in loop | n |
|---|---|---|---|---|---|---|---|
| **MemBukkit** (`original_openai_depth50_fullscan`) | `text-embedding-3-large@1536` (gbrain's exact config) | **99.40%** (497/500) | 92.40 | 98.20 | 99.80 | no | 500 |
| gbrain-hybrid ([report](https://github.com/garrytan/gbrain-evals/blob/main/docs/benchmarks/2026-05-07-longmemeval-s.md)) | `text-embedding-3-large@1536` + BM25 + RRF | 97.60% (488/500) | | | | no | 500 |
| gbrain-vector | `text-embedding-3-large@1536` | 97.40% | | | | no | 500 |
| MemPalace raw (ChromaDB) | | 96.6% | | | | no | 500 |
| **MemBukkit** (`original_local_depth50_fullscan`) | `biencoder_v1` (our 110M fine-tuned mpnet, laptop CPU/MPS, $0 API) | 96.40% (482/500) | 86.20 | 94.60 | 98.60 | no | 500 |

gbrain and MemPalace numbers are theirs, copied from their published reports; we
did not rerun their systems. Ours are reproduced from this repo with the commands
below, and the per-question rankings are checked in under `results/`.

Per-type breakdown for the `text-embedding-3-large` arm (any-gold@5 / all-gold@5):

| type | n | any@5 | all@5 |
|---|---|---|---|
| knowledge-update | 78 | 100.00 | 98.72 |
| multi-session | 133 | 100.00 | 82.71 |
| single-session-assistant | 56 | 100.00 | 100.00 |
| single-session-preference | 30 | 96.67 | 96.67 |
| single-session-user | 70 | 98.57 | 98.57 |
| temporal-reasoning | 133 | 99.25 | 83.46 |

`all-gold@k` (every gold session in the top-k) is the stricter multi-session
number nobody publishes; it is here because it is the one that actually predicts
whether a reader can answer a multi-session question from the retrieved context.

The three misses at K=5: `0862e8bf_abs` (single-session-user, abstention),
`d6233ab6` (single-session-preference), `4dfccbf8` (temporal-reasoning). Two of
them land at rank 6-10.

## What the comparison does and does not show

- **Same encoder, better stack.** The headline arm uses gbrain's exact embedding
  model, so the +1.8 over gbrain-hybrid and +2.0 over gbrain-vector is the
  retrieval layer on top: turn-level facts instead of chunks, and a 22M
  cross-encoder fused with cosine by RRF. The cross-encoder is
  `cross-encoder/ms-marco-MiniLM-L-6-v2` fine-tuned on LongMemEval-style
  query/fact pairs (see [`docs/METHOD.md`](../../docs/METHOD.md)); it runs
  locally and costs nothing per query. Query cost is therefore the same as
  gbrain's: one embedding call, roughly $0.50 per 1000 questions.
- **Our own encoder is below gbrain.** The zero-API arm swaps in our 110M
  fine-tuned bi-encoder: 96.4, 1.2 behind gbrain-hybrid and a statistical tie
  with MemPalace raw (96.6). It is a laptop model competing against a hosted
  embedding model, and the recall gap is what you would expect. It is
  published here rather than hidden because the point of MemBukkit is not
  "we have a better embedding model"; it is that the layer above the encoder
  is where the recall comes from, and it works on top of whichever encoder you
  can afford.
- **The reranker is conditional, and this is a case where it helps.** Across our
  other benchmarks the cross-encoder helps on weak encoders and hurts or ties on
  strong ones ([`../PAPER_RESULTS.md`](../PAPER_RESULTS.md),
  [`../beam_ablation/`](../beam_ablation/)). Session-level recall on `_s`
  haystacks (~50 sessions, ~500 turns) is a small opened region where a
  hard-negative-trained cross-encoder has room to lift the tail. Do not read
  this table as evidence that the reranker helps at 10M-token scales; the BEAM
  ablation says otherwise.
- **Two harness arms are on, and they are harness fixes, not retrieval changes.**
  `--scan-budget 1.0` disables topic-bucket gating so we search the whole store,
  exactly like gbrain does (production runs at ~33% scan to cut reader tokens).
  `--rank-depth 50` derives the session ranking from the top-50 reranked facts
  instead of the production `top_k=10` cut; with a 10-fact cut, facts often
  collapse onto fewer than five distinct sessions, so recall@5 would be scored
  against an undersized list. The "as shipped" configuration (33% scan, top_k=10)
  with the local encoder scores 95.4 on this metric, and it is the configuration
  that produces the [82.0 QA accuracy](../../README.md) with a gpt-4o-mini reader
  on 3.2k tokens of context. The two arms measure retrieval; the production cut
  is tuned for answer accuracy per token.
- **Recall@5 is the easy half of the problem.** gbrain's own report says
  vector-only lands at 97.4, 0.2 behind their full hybrid. On this split, most of
  the top-5 recall is the encoder. What separates memory systems is what happens
  after retrieval: how few tokens you can hand the reader and still answer.
  That is why our headline number in the main README is QA accuracy at a fixed
  token budget, not this one.

## Reproduce

Both commands run on the original HuggingFace split, download it on first use,
and resume from `per_question.jsonl` if interrupted. Embeddings are cached in
`.membukkit_emb_cache/` keyed by sha256(text); a cold run of the OpenAI arm costs
about $2 in embeddings, a warm run costs nothing. Roughly 10 minutes warm on an
M-series laptop for the OpenAI arm, about 65 minutes for the local-encoder arm
(local encoding of every haystack turn dominates).

```bash
# gbrain-matched arm: their encoder, our stack, full scan, depth-50 ranking
OPENAI_API_KEY=... uv run python scripts/session_recall.py \
    --encoder openai:text-embedding-3-large@1536 --rank-depth 50 --scan-budget 1.0 \
    --output benchmarks/longmemeval_session_recall/results/original_openai_depth50_fullscan

# zero-API arm: our 110M fine-tuned local encoder (auto-downloaded from the Hub)
uv run python scripts/session_recall.py --rank-depth 50 --scan-budget 1.0 \
    --output benchmarks/longmemeval_session_recall/results/original_local_depth50_fullscan

# as-shipped production cut (33% scan, top_k=10), for the honest lower bound
uv run python scripts/session_recall.py \
    --output benchmarks/longmemeval_session_recall/results/original_local_production
```

`summary.json` carries the aggregate and per-type numbers; `per_question.jsonl`
carries the ranked session ids and gold ids for every question so any row above
can be re-scored without rerunning anything.
