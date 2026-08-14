# Storage backends

`MemorySystem` talks to a storage backend through a small protocol
([`storage/base.py`](../code-walkthrough.md)), so the persistence layer is swappable.

| Backend | Module | Use for |
|---------|--------|---------|
| In-memory | `membukkit.storage.memory.InMemoryBackend` | Tests, notebooks, single-process evaluation. Nothing persists. |
| Turbopuffer | `membukkit.storage.turbopuffer.TurbopufferBackend` | Production. One namespace per memory owner, persistent vectors + metadata. |

Select the backend via `StorageConfig`:

```python
from membukkit import MemorySystem, ModelConfig, RetrievalConfig
from membukkit.config import StorageConfig

mem = MemorySystem.from_pretrained(
    models=ModelConfig(model_dir="./models"),
    retrieval=RetrievalConfig(),
    llm="openai:gpt-4o-mini",
    storage=StorageConfig(
        backend="turbopuffer",
        namespace="my-namespace",
        region="gcp-us-central1",         # or env TURBOPUFFER_REGION
        # api_key defaults to env TURBOPUFFER_API_KEY
        vector_dtype="f16",
    ),
)
```

## Turbopuffer layout

Each memory owner maps to a single Turbopuffer namespace. Facts are stored with their embedding
(f16 by default) plus filterable metadata (`ts`, `topic_bucket`, `entities`, `subject`,
`superseded_by`, …). Retrieval is a single `multi_query` (dense ANN lane + optional BM25 lexical
lane) fused server-side via RRF, after which the local cross-encoder reranks the pool.

The topic partition (KMeans centroids) is persisted as a sentinel `__partition__` document *inside
the same namespace* and cached per worker.

> **The `payload` attribute is not filterable**
>
> The centroids blob (`payload`, ~96 KiB) is stored with `filterable: False`. Turbopuffer caps
> **filterable** attributes at 4 KiB, so a filterable `payload` would be rejected.

## The topic partition

The partition is built lazily. `mem.partition()`:

1. Loads the persisted `__partition__` doc if present.
2. Otherwise **samples** fact vectors, runs KMeans (`num_buckets`), persists the centroids, and
   returns `{k_eff, version, sizes, ...}`.

`ingest_facts` and `ingest` write facts with `topic_bucket = -1` until a partition exists, the
bucket assignment happens against the *cached* centroids at write time, or during a re-cluster.

> **Facts ingested before a partition exists are unlabeled**
>
> If you bulk-load with `ingest_facts` and never build/recluster, every fact keeps
> `topic_bucket = -1`. That is fine for **open** retrieval (the ANN spans the whole bank) but
> those facts are invisible to **gated** retrieval and won't appear in the bucket trace. Run a
> re-cluster to label them. See
> [Troubleshooting → empty `opened_buckets`](troubleshooting.md#opened_buckets-is-always-empty).

## Re-clustering

`recluster()` rebuilds centroids from a sample and then **re-labels every live fact's**
`topic_bucket` (a stream of scalar patches, vectors are never re-written). Queries keep using the
old partition until the new centroids are saved atomically.

```python
mem._backend.recluster()          # full rebuild + relabel of all facts
mem._backend.maybe_recluster()    # only if the bank grew past the threshold
```

The service exposes `maybe_recluster()` at `POST /v1/{owner}/recluster` (see
[Memory service](service.md)). To force a full relabel of an existing bank, call `recluster()`
directly.

## Scanning within Turbopuffer's limits

Turbopuffer caps a single query's `top_k` (ceiling `_MAX_TOP_K = 1200`), and its query API has no
cursor/offset. Full-bank scans (partition sampling, re-cluster relabel) therefore **keyset-paginate**:
order by `id` ascending and advance each page with an `id > last_seen` filter. This walks the entire
bank with no duplicates or skips, one `≤ 1200`-row page at a time.

This matters operationally:

- `_build_partition` computes centroids from **all** live facts (paged), not a truncated sample.
- `recluster` re-labels **every** fact, regardless of bank size.

> **Historical note**
>
> Earlier the backend issued a single `top_k=50_000` query (→ HTTP 400) and relied on a
> non-existent response cursor, which silently capped scans at one page. Both are fixed; the
> debugging trail is preserved in [Troubleshooting](troubleshooting.md).

## Gated vs. open retrieval

`RetrievalConfig.retrieval_mode` (also `--retrieval-mode` on the service) controls how buckets are
used at query time:

| Mode | Behavior |
|------|----------|
| `gated` (default) | Open buckets until the scan budget is met, then query **only** those buckets. Sublinear, interpretable: but requires facts to be bucket-labeled. |
| `open` | ANN spans the whole bank; buckets are derived **post-hoc** for the trace only. Robust when facts are unlabeled, at the cost of the scan-budget savings. |

If you bulk-loaded facts and haven't re-clustered, use `open` (or re-cluster first, then use
`gated`).
