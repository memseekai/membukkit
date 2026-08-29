# MemBukkit vs QMD — full retrieval results

All runs: **1,000 questions**, identical corpora, identical `queries.json`, scored by **QMD's own scorer** (`score.ts` port), 10 results per query.

**The embedder is stock in every row.** Only two components were trained here: a 22M cross-encoder reranker (`RR`) and a ~1k-parameter fusion MLP (`MLP`).

## Best configuration per stage (`chain` mode)

| Stage | Trained | 2Wiki prec | MuSiQue prec |
|---|---|---:|---:|
| baseline — MPNet + reranker_v2 | — | 0.716 | *superseded* |
| + Qwen3 encoder swap (stock) | — | 0.842 | 0.700 |
| + tail-boosted reranker | RR | 0.866 | **0.722** |
| **+ anchored fusion** | **RR+MLP** | **0.880** | 0.718 |
| QMD best backend | their stack | 0.749 | 0.670 |

## 2Wiki — 1,000 q / 6,119 passages

| | Config | Stack | Fusion | Trained | Mode | prec | R@1 | R@3 | R@5 | MRR | latency |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
|  | baseline | MPNet 110M + reranker_v2 | RRF | — | dense | 0.613 | 0.299 | 0.481 | 0.542 | 0.773 | 14ms |
|  |  |  |  |  | rerank | 0.673 | 0.369 | 0.574 | 0.618 | 0.896 | 90ms |
|  |  |  |  |  | chain | 0.716 | 0.369 | 0.608 | 0.657 | 0.892 | 236ms |
|  | encoder swap | Qwen3-0.6B + reranker_v2 | RRF | — | dense | 0.721 | 0.427 | 0.661 | 0.689 | 0.983 | 25ms |
|  |  |  |  |  | rerank | 0.731 | 0.433 | 0.674 | 0.699 | 0.990 | 151ms |
|  |  |  |  |  | chain | 0.842 | 0.433 | 0.740 | 0.795 | 0.989 | 291ms |
|  | LLM decompose | Qwen3-0.6B + reranker_v2 | RRF | — | decompose | 0.857 | 0.433 | 0.736 | 0.807 | 0.989 | 2453ms |
| ✗ | rr IIRC+QASC | Qwen3-0.6B + armA flat | RRF | **RR** | rerank | 0.723 | 0.407 | 0.639 | 0.680 | 0.957 | 189ms |
|  |  |  |  |  | chain | 0.838 | 0.407 | 0.701 | 0.771 | 0.955 | 361ms |
| ✗ | rr IIRC only | Qwen3-0.6B + IIRC flat | RRF | **RR** | rerank | 0.728 | 0.407 | 0.639 | 0.683 | 0.958 | 118ms |
|  |  |  |  |  | chain | 0.802 | 0.407 | 0.653 | 0.726 | 0.954 | 261ms |
| ✗ | rr HotpotQA | Qwen3-0.6B + Hotpot flat | RRF | **RR** | rerank | 0.704 | 0.368 | 0.614 | 0.651 | 0.907 | 118ms |
|  |  |  |  |  | chain | 0.796 | 0.368 | 0.643 | 0.716 | 0.905 | 293ms |
|  | rr TAIL-BOOST | Qwen3-0.6B + tail-boosted | RRF | **RR** | rerank | 0.739 | 0.435 | 0.680 | 0.707 | 0.992 | 112ms |
|  |  |  |  |  | chain | 0.866 | 0.435 | 0.771 | 0.826 | 0.992 | 257ms |
|  | anchored fusion | Qwen3-0.6B + reranker_v2 | anchored | **MLP** | rerank | 0.729 | 0.432 | 0.674 | 0.704 | 0.989 | 111ms |
|  |  |  |  |  | chain | 0.845 | 0.432 | 0.727 | 0.787 | 0.988 | 253ms |
| ★ | BEST | Qwen3-0.6B + tail-boosted | anchored | **RR+MLP** | rerank | 0.747 | 0.440 | 0.690 | 0.715 | 0.998 | 173ms |
|  |  |  |  |  | chain | 0.880 | 0.440 | 0.814 | 0.862 | 0.998 | 342ms |
|  | QMD | EmbeddingGemma-300M + Qwen3-Reranker | their stack | n/a | bm25 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 5ms |
|  |  |  |  |  | vector | 0.742 | 0.421 | 0.667 | 0.704 | 0.974 | 163ms |
|  |  |  |  |  | hybrid | 0.729 | 0.396 | 0.642 | 0.689 | 0.941 | 2313ms |
|  |  |  |  |  | full | 0.749 | 0.396 | 0.676 | 0.712 | 0.944 | 3112ms |

## MuSiQue — 1,000 q / 11,656 passages

| | Config | Stack | Fusion | Trained | Mode | prec | R@1 | R@3 | R@5 | MRR | latency |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
|  | encoder swap | Qwen3-0.6B + reranker_v2 | RRF | — | dense | 0.628 | 0.289 | 0.471 | 0.540 | 0.794 | 30ms |
|  |  |  |  |  | rerank | 0.657 | 0.310 | 0.503 | 0.578 | 0.837 | 148ms |
|  |  |  |  |  | chain | 0.700 | 0.310 | 0.532 | 0.609 | 0.823 | 281ms |
|  | LLM decompose | Qwen3-0.6B + reranker_v2 | RRF | — | decompose | 0.739 | 0.310 | 0.562 | 0.648 | 0.834 | 2150ms |
| ✗ | rr IIRC+QASC | Qwen3-0.6B + armA flat | RRF | **RR** | rerank | 0.609 | 0.247 | 0.430 | 0.518 | 0.708 | 127ms |
|  |  |  |  |  | chain | 0.650 | 0.247 | 0.441 | 0.541 | 0.697 | 275ms |
| ✗ | rr IIRC only | Qwen3-0.6B + IIRC flat | RRF | **RR** | rerank | 0.586 | 0.232 | 0.395 | 0.472 | 0.667 | 126ms |
|  |  |  |  |  | chain | 0.578 | 0.232 | 0.379 | 0.455 | 0.630 | 268ms |
| ✗ | rr HotpotQA | Qwen3-0.6B + Hotpot flat | RRF | **RR** | rerank | 0.567 | 0.167 | 0.331 | 0.432 | 0.561 | 128ms |
|  |  |  |  |  | chain | 0.565 | 0.167 | 0.332 | 0.432 | 0.551 | 254ms |
|  | rr TAIL-BOOST | Qwen3-0.6B + tail-boosted | RRF | **RR** | rerank | 0.663 | 0.309 | 0.519 | 0.591 | 0.835 | 121ms |
|  |  |  |  |  | chain | 0.722 | 0.309 | 0.542 | 0.626 | 0.817 | 249ms |
|  | anchored fusion | Qwen3-0.6B + reranker_v2 | anchored | **MLP** | rerank | 0.645 | 0.319 | 0.503 | 0.571 | 0.855 | 125ms |
|  |  |  |  |  | chain | 0.710 | 0.319 | 0.534 | 0.616 | 0.852 | 255ms |
| ★ | BEST | Qwen3-0.6B + tail-boosted | anchored | **RR+MLP** | rerank | 0.659 | 0.307 | 0.508 | 0.586 | 0.827 | 154ms |
|  |  |  |  |  | chain | 0.718 | 0.307 | 0.552 | 0.635 | 0.820 | 340ms |
|  | QMD | EmbeddingGemma-300M + Qwen3-Reranker | their stack | n/a | bm25 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 13ms |
|  |  |  |  |  | vector | 0.666 | 0.312 | 0.501 | 0.579 | 0.831 | 402ms |
|  |  |  |  |  | hybrid | 0.615 | 0.269 | 0.457 | 0.530 | 0.754 | 3640ms |
|  |  |  |  |  | full | 0.670 | 0.269 | 0.520 | 0.601 | 0.781 | 4305ms |

★ = best · ✗ = regressed below the shipped reranker

## Significance, best config vs QMD best-per-metric

| Metric | 2Wiki Δ | σ | MuSiQue Δ | σ |
|---|---:|---:|---:|---:|
| precision | +0.131 | **+7.6** | +0.048 | **+2.3** |
| R@3 | +0.138 | **+7.2** | +0.032 | +1.4 |
| R@5 | +0.150 | **+8.3** | +0.034 | +1.6 |
| R@1 | +0.019 | +0.9 | −0.005 | −0.2 |

## Notes

- **The three ✗ rows are the informative failures.** On MuSiQue all three flat-trained rerankers fall *below plain dense* (0.628). Same data pipelines, same encoder, same fusion — the only difference from the 0.722 tail-boosted row is whether positives were cosine-rank-ordered with the tail oversampled.
- **R@1 is at its ceiling on 2Wiki.** With 2–4 gold passages and one slot, max R@1 is 0.441; we score 0.440. That column cannot move. MuSiQue's ceiling is 0.406 against our 0.307, so it retains ~10 points of headroom and stays a tie with QMD.
- **MuSiQue's best is split**: LLM `decompose` leads on precision/R@3/R@5 (0.739 / 0.562 / 0.648) but costs 2,150ms against 249ms for tail-boost `chain`.
- **The MuSiQue MPNet baseline is absent** — that run used buggy gold labels and is superseded, so there is no valid before-number for MuSiQue.
- **QMD's bm25 rows are a configuration artifact** of long natural-language queries against FTS5 implicit-AND, not a real QMD weakness.
- Latency is single-query mean and varies with machine load; treat as indicative.
