# MEMBUKKIT Method

## Overview

MEMBUKKIT implements a **topic-bucket-gated hybrid retrieval** system for long-term conversational memory. The method scans ~33% of memory while matching or exceeding full-scan accuracy.

## Pipeline

### 1. Distillation (Ingest)
Raw conversation turns are distilled into **atomic, self-contained, dated facts** via an LLM (gpt-4o-mini). Each fact is a single declarative sentence with resolved coreference and preserved specifics.

- Prompt: `src/membukkit/prompts/extraction.py`
- Implementation: `src/membukkit/extraction/distiller.py`
- Cache: content-hash keyed, so repeated calls cost nothing

### 2. Embedding + Partitioning
Atomic facts are encoded with a fine-tuned bi-encoder (E16, `all-mpnet-base-v2` base, contrastive fine-tuned on PerLTQA). The embedding space is partitioned into **K topic buckets** via KMeans.

- Encoder: `src/membukkit/models/encoder.py`
- Buckets: `src/membukkit/retrieval/buckets.py` → `build_topic_partition()`

### 3. Routing
A query is mapped to the partition: cosine(query_emb, centroid) gives a soft probability per bucket. Buckets are opened in descending probability until the **scan budget** (default 30%) is met.

- Router: `src/membukkit/retrieval/buckets.py` → `route_topic()`
- Multi-axis variant: topic + entity + time axes unioned (`route_multiaxis()`)

### 4. Reranking
Within the opened region, a fine-tuned **cross-encoder** (C1, `ms-marco-MiniLM-L-6-v2` base) scores each `(query, fact)` pair for relevance.

- Reranker: `src/membukkit/models/reranker.py` → `UtilityReranker`

### 5. Fusion (RRF)
The **shipped "hybrid" mode** fuses cross-encoder and cosine rankings via Reciprocal Rank Fusion:
```
RRF_score(i) = 1/(k + rank_xenc(i)) + 1/(k + rank_cosine(i))
```
This catches both semantic (cosine) and relevance (cross-encoder) signals.

- Implementation: `src/membukkit/retrieval/buckets.py` → `rrf_order()`

### 6. Reading
The top-K facts are passed to a **query-type-routed reader**:
- **Dated reader**: single-fact lookups (default)
- **Reasoning reader**: multi-fact synthesis (temporal, aggregation, knowledge-update)
- **Recommendation reader**: preference/advice queries

Query-type routing is rule-based and text-only (no gold labels → leakage-free).

- Router: `src/membukkit/retrieval/router.py`
- Readers: `src/membukkit/reading/readers.py`

## Key Design Decisions

1. **Buckets for retrieval, not just interpretability**: KMeans clusters ARE the retrieval index.
2. **Scan budget, not fixed K**: open buckets until N% of memory is scanned (adaptive to bank size).
3. **Hybrid RRF > pure cross-encoder**: the cross-encoder misses some semantic matches cosine catches.
4. **Dual-index union**: verbatim turns + atomic facts, merged by cross-encoder score.
5. **Query-conditioned depth**: reasoning queries get deeper retrieval (top-30 vs top-10).
