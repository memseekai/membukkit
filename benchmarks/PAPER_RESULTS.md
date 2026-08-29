# Research results behind the retrieval design

These are results from the CoreMem research that produced MemBukkit's retrieval
stack. **They were not measured by the benchmark suite in this directory** and
are not reproducible with it: they use a different reader, encoder, and metric.
They are recorded here because they explain *why* the retriever is built the way
it is, and because several of them are negative results worth knowing before you
try the same ideas.

Read the two sources separately:

|  | [`README.md`](README.md) (this suite) | This file (research) |
|---|---|---|
| Measures | retrieval only, no LLM | retrieval + a reader |
| Metric | any/all-support Recall@k, MRR, nDCG | EM / F1 / Recall@5 |
| Encoder | shipped MPNet bi-encoder | swept: MPNet, Qwen3, NV-Embed-v2 |
| Reader | none | `gpt-4o-mini`, temp 0, top-5 |
| Corpora | QMD fixture, HotpotQA, MuSiQue, 2Wiki | MuSiQue, 2Wiki, HotpotQA |

`R@5` below is **fraction of gold passages found**, which is not the same as this
suite's strict `All@5` (*every* gold document retrieved). Directions transfer
between the two; absolute values do not.

---

## 1. Encoder strength dominates, and it inverts the value of reranking

Mean CoreMem − Dense across the three datasets, per encoder. The same encoder
drives both arms, so this isolates what the retrieval machinery adds:

| Encoder | Params | Δ EM | Δ F1 | Δ R@5 |
|---|---:|---:|---:|---:|
| all-mpnet-base-v2 | 110M | **+3.4** | **+3.5** | **+4.2** |
| Qwen3-Embedding-0.6B | 600M | +0.3 | +0.6 | −0.3 |
| NV-Embed-v2 | 7B | **−3.4** | **−3.4** | **−5.7** |

The gating, reranking, and fusion stack **helps a weak encoder and hurts a strong
one**, monotonically across all three metrics. This is the single most useful
thing to know before tuning retrieval: the right stack depends on the encoder.

## 2. A bigger reranker does not fix it

Isolating the within-region ranker on MuSiQue with NV-Embed-v2 and the gate
fully open:

| Reranker | Fusion | EM | F1 | R@5 |
|---|---|---:|---:|---:|
| none (cosine only) | cosine | 35.2 | 45.2 | 71.8 |
| MiniLM-L6 (22M) | rerank | 27.2 | 35.5 | 57.0 |
| MiniLM-L6 (22M) | RRF | 31.6 | 41.4 | 65.4 |
| bge-reranker-v2-m3 (568M) | rerank | 28.4 | 38.1 | 64.0 |
| bge-reranker-v2-m3 (568M) | RRF | 35.9 | 45.9 | 71.0 |
| dense (reference) | — | 35.8 | 45.7 | 71.8 |

A 568M cross-encoder reranking a 7B encoder's top-100 still lands **7 R@5 points
below plain cosine**. The best the stack manages is a tie, and only by deferring
to cosine through RRF. "Use a bigger reranker" is a measured dead end.

## 3. Iterative decomposition is what beats dense

Single-shot retrieve-then-rerank cannot recover a second-hop passage that is
dissimilar to the original query. Adding a training-free, graph-free iterative
loop does:

| Config | MuSiQue (EM/F1/R@5) | 2Wiki (EM/F1/R@5) | HotpotQA (EM/F1/R@5)¹ |
|---|---|---|---|
| dense (NV-Embed) | 35.8 / 45.7 / 71.8 | 54.9 / 60.0 / 77.2 | 61.5 / 73.9 / 93.9 |
| CoreMem single-shot | 31.8 / 41.2 / 64.5 | 50.0 / 55.5 / 70.9 | — |
| soft hop-2 (entity-expanded) | 36.3 / 46.3 / 72.6 | 56.7 / 62.6 / 81.2 | — |
| **iterative decompose** | **42.5 / 52.1 / 79.7** | **63.9 / 71.8 / 92.6** | **63.9 / 75.7 / 96.3** |

### The two fixes are the entire result

Naive decomposition (sub-questions retrieved in parallel, fused by pooling)
scores *below* dense. Toggling only iterative bridge substitution and interleave
fusion:

| Dataset | naive (parallel + pool) | fixed (iterative + interleave) | Δ |
|---|---|---|---|
| MuSiQue | 32.0 / 41.8 / 69.3 | **42.5 / 52.1 / 79.7** | +10.5 / +10.3 / +10.4 |
| 2Wiki | 53.3 / 58.1 / 76.1 | **63.9 / 71.8 / 92.6** | +10.6 / +13.7 / +16.5 |

Decomposition is not automatically good. Done naively it actively hurts, for two
identifiable reasons: an unresolved bridge reference ("awards of *[the director
of X]*") retrieves noise, and uncalibrated cross-query pooling lets one
sub-question evict another's gold passage. `RagRetriever` implements the fixed
form, and [`rag.py`](../src/membukkit/retrieval/rag.py) documents why.

## 4. Negative results worth not repeating

- **Answer self-verification hurts.** A verification pass left R@5 unchanged and
  cut EM by 6–9 points. Self-correction without external feedback over-edits
  answers that were already right. Disabled by default.
- **Three hops are worse than two**, and entity-only expansion beats appending
  passage text.
- **The entity axis is dataset-conditional.** It lifts 2Wiki (entity-comparison
  questions) by up to +6 R@5 and *hurts* MuSiQue (long chains, noisy entity
  sets) at every threshold tested.
- **A temporal axis had no effect** on Wikipedia QA, which is not time-gated.
  Its value shows up on dated conversational benchmarks instead.

## 5. Published comparison

`gpt-4o-mini` reader, NV-Embed-v2 embedder, top-5, EM/F1:

| Dataset | CoreMem (decompose) | Dense-NV | HippoRAG 2 | MultiCube-RAG |
|---|---|---|---|---|
| MuSiQue | **42.5 / 52.1** | 32.8 / 46.0 | 35.0 / 49.3 | 39.5 / 50.9 |
| 2Wiki | **63.9 / 71.8** | 54.4 / 60.8 | 60.5 / 69.7 | 63.2 / 71.5 |
| HotpotQA | **63.9 / 75.7**¹ | 57.3 / 71.0 | 56.3 / 71.1 | 57.5 / 71.5 |

Caveats carried over from the research, unchanged:

- ¹ **The HotpotQA row is indicative, not like-for-like.** That corpus is easier
  than the baselines' full-corpus setting, and the *dense* baseline is inflated
  there too. The decompose-over-dense gain on the same corpus is valid; the
  absolute SOTA delta is not.
- MuSiQue and 2Wiki *are* calibrated: the dense baseline matches published dense
  numbers, so the gain is measured from a trustworthy floor.
- All rows are 1,000-question samples.
- The winning config spends query-time LLM (one split plus up to `max_subq`
  short-answer calls). That is the same cost class as the graph-based baselines,
  which also decompose. The difference is index time: **zero index-time LLM
  calls**, against a corpus-wide LLM indexing pass for the others.

## Relationship to this suite

The suite in this directory measures retrieval alone, with no reader and no
judge, so it answers a narrower question and answers it more cheaply. Where the
two overlap they agree: on this repo's own HotpotQA measurements, entity-chain
expansion beat single-shot retrieval, pooling lost to interleaving, two hops beat
three, and entity-only expansion beat appending passage text — all reproduced
independently before this file was written.
