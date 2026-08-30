# BEAM 100K: does the cross-encoder earn its keep at scale?

Paired ablation of `--rerank-select hybrid` (shipped: RRF over cosine +
cross-encoder ranks) vs `--rerank-select none` (plain cosine inside the same
opened region, cross-encoder never runs), on the BEAM 100K split: 20
conversations, 400 probing questions, scored by BEAM's own vendored judge
(gpt-4.1-mini @ temp 0, official prompts). Within each pair everything except
`--rerank-select` is identical, including the distill cache, so ingestion is
bit-for-bit the same.

Run on 2026-08-29/30. Motivated by
[this r/Rag question](https://www.reddit.com/r/Rag/): "does the reranker earn
more at BEAM scales, where the opened region is much bigger?"

## Results (official BEAM average)

| encoder | reader | hybrid | none | drop the CE |
|---|---|---:|---:|---:|
| text-embedding-3-large@1536 | gpt-4o-mini | 0.518 | 0.546 | **+2.9** |
| text-embedding-3-large@1536 | Gemma 4 26B | 0.528 | 0.539 | **+1.1** |
| all-mpnet-base-v2 (weak) | gpt-4o-mini | 0.544 | 0.531 | **−1.3** |

Answer: **no — on the shipped strong-encoder config the reranker earns *less*
at scale, not more.** Dropping it entirely gains 1–3 points with both readers.
The weak-encoder row is the crossover: there the cross-encoder still pays
(+1.3), consistent with the LongMemEval and RAG ablations
(`benchmarks/PAPER_RESULTS.md`).

The Gemma hybrid arm doubles as an independent reproduction of the frozen
`beam-100k-gemma` recipe: measured 0.528 vs expected 0.535, PASS within the
±0.02 band (`membukkit bench --repro beam-100k-gemma --check`).

## The matched-CE control (`tel3l-hybrid/`)

The shipped reranker's hard negatives were mined with an mpnet-family
bi-encoder, so next to text-embedding-3-large it is *tail-mismatched*. To
separate "wrong tail" from "no tail", we retrained the same 22M
ms-marco-MiniLM CE with the identical recipe (`scripts/train_reranker.py`,
80 LME train banks / 40 held-out, same negatives, same epochs) but mined the
hard negatives with text-embedding-3-large itself
(`--encoder openai:text-embedding-3-large@1536`). Trained on LongMemEval,
evaluated on BEAM: no contamination.

| arm (3-large, gpt-4o-mini) | average |
|---|---:|
| shipped CE (mpnet-mined tail) | 0.518 |
| matched CE (3-large-mined tail) | 0.534 |
| no CE | 0.546 |

Matched training recovers about half the gap and still loses to plain cosine.
Held-out diagnostics explain why (`tel3l-reranker-heldout-eval.json`): with
text-embedding-3-large, cosine hit@10 on the training distribution is 1.000 —
there is no miss-tail left to rescue. This reproduces the paper's LME ordering
(OOD CE < in-domain CE < plain cosine) at BEAM scale.

Consistent detail across the strong-encoder rows: `information_extraction` is
where plain cosine wins biggest (+10 to +13 points) — the CE's
"sounds-like-the-query" bias costs the most on precise lookups.

## Caveats

- One run per arm, n=400, stochastic readers and judge: per-category deltas
  are noisy; the cross-arm *direction flips* are the finding, not any single
  point estimate.
- Readers here are gpt-4o-mini and Gemma 4 26B (the frozen-recipe reader). The
  gpt-4o-mini absolute numbers are not comparable to the published 0.535,
  which used Gemma; only arms within a row are comparable.
- BEAM's `abstention` category moves in opposite directions across readers
  (hybrid helps 4o-mini abstain, hurts Gemma): treat that category as
  reader-coupled.

## Reproduce

```bash
# shared flags for every arm below
COMMON="--dataset beam --beam-scale 100K --methods coremem_union \
  --bucket-mode topic --reader gpt-4o-mini --judge gpt-4o \
  --reader-prompts v1 --agg-top-k 50 --agg-scan-budget 1.0 \
  --deep-top-k 60 --deep-scan-budget 1.0 --deep-broad \
  --distill-model gpt-4o-mini --distill-cache runs/distill_beam100k.json"
export MEMBUKKIT_DISTILL_MAX_TURN_CHARS=4000

# strong-encoder pair
membukkit eval $COMMON --encoder openai:text-embedding-3-large@1536 \
  --rerank-select hybrid --output-dir results/ablation-hybrid
membukkit eval $COMMON --encoder openai:text-embedding-3-large@1536 \
  --rerank-select none --output-dir results/ablation-none

# weak-encoder pair
membukkit eval $COMMON --encoder sentence-transformers/all-mpnet-base-v2 \
  --rerank-select hybrid --output-dir results/ablation-mpnet-hybrid
membukkit eval $COMMON --encoder sentence-transformers/all-mpnet-base-v2 \
  --rerank-select none --output-dir results/ablation-mpnet-none

# matched-CE control: train, then evaluate
python scripts/train_reranker.py \
  --encoder openai:text-embedding-3-large@1536 --output-dir models/reranker_tel3l
membukkit eval $COMMON --encoder openai:text-embedding-3-large@1536 \
  --reranker models/reranker_tel3l/model \
  --rerank-select hybrid --output-dir results/ablation-tel3l-hybrid
```

The Gemma arms swap `--reader`/`--distill-model` for
`compat:google/gemma-4-26b-a4b-it` (any OpenAI-compatible host; we used
DeepInfra) with their own distill cache, per the `beam-100k-gemma` recipe in
`src/membukkit/bench/recipes.py`.

Each `<arm>/` directory holds the run's `beam_summary.json` (official
per-category scores) and `e2e_summary.json` (harness config + acc@0.7 view).
