# Troubleshooting

A field guide to the Turbopuffer + service issues most likely to bite, with the exact diagnostics
to tell them apart. Most symptoms below reduce to one of: **wrong namespace**, **unlabeled facts**,
or **a query that exceeds a Turbopuffer limit**.

## Diagnostic: count a namespace directly

Almost every issue starts by confirming *what is actually in the namespace* the server talks to.
This mirrors exactly what the backend does:

```python
import os
from membukkit.config import RetrievalConfig, StorageConfig
from membukkit.storage.turbopuffer import TurbopufferBackend

s = StorageConfig(backend="turbopuffer", namespace="my-namespace",
                  region=os.environ.get("TURBOPUFFER_REGION"))
b = TurbopufferBackend(RetrievalConfig(), None, s)   # encoder=None is fine for count()
print("count:", b.count())
```

Run it in the **same shell** you start the server from, so it inherits the same
`TURBOPUFFER_REGION` / `TURBOPUFFER_API_KEY`.

## `/partition` returns `{k_eff: 0, sizes: {}}`

`sizes: {}` means the backend found **no fact vectors to cluster**. Work through, in order:

1. **Wrong namespace (prefix).** The service maps owner → `<prefix><owner>` (default `mem_`). A
   request to `/v1/acme-team-1/...` hits `mem_acme-team-1`, not `acme-team-1`. Count both the
   literal and the prefixed name. Fix by starting the server with `--namespace-prefix ""` (see
   [Memory service](service.md#owner-namespace-mapping)) or by using the correct owner id.

2. **Region / credential mismatch.** Turbopuffer namespaces are scoped to a region+project. If the
   ingest shell and the server shell had different (or unset) `TURBOPUFFER_REGION`, the server is
   looking at a *different, empty* namespace of the same name. Set the region explicitly in both.

3. **The ingest never completed.** If a `from_pretrained` run crashed at model load, nothing was
   written. Re-run to completion and watch for the `namespace now holds N facts` line.

4. **A partition-build query hit a limit**, historically a `top_k=50_000` sample query returned
   HTTP 400 and was swallowed, leaving the partition empty. Fixed by keyset pagination (see below).

## `opened_buckets` is always empty

Symptom: `/answer` works and returns correct facts, but `trace.opened_buckets` is `[]`.

Cause: every fact has `topic_bucket = -1`. Facts written by `ingest_facts`/`ingest` **before a
partition existed** are never assigned to a bucket, and `_posthoc_buckets` filters out `-1`. In
**open** mode this is cosmetic (the ANN spans the whole bank, so answers are correct); in **gated**
mode it would hide those facts entirely.

Fix, re-cluster to label every fact, then buckets populate and gated mode works:

```python
b.recluster()   # rebuilds centroids from ALL facts (keyset-paged) and relabels every row
```

The `POST /v1/{owner}/recluster` endpoint calls `maybe_recluster()`, which no-ops when the bank
hasn't grown, call `recluster()` directly (as above) to force a full relabel.

## Turbopuffer 400: `attribute "payload" not found in include_attributes`

Raised by `_load_partition` on a namespace that has **no partition doc yet**, the `payload` column
doesn't exist until the first partition is saved. The backend requests `include_attributes=True`
(not `["payload"]`) so this no longer 400s; it simply returns "no partition" and triggers a build.

## Turbopuffer 400: `payload value too large for filtering, limit is 4096 bytes`

The centroids blob (~96 KiB) was being declared as a **filterable** string. Filterable attributes
are capped at 4 KiB. The partition doc now declares `payload` with `filterable: False`, stored, not
indexed.

## Turbopuffer 400 on large scans / `top_k`

Turbopuffer caps a single query's `top_k` (ceiling 1200) and offers **no cursor/offset**. A naive
`top_k=50_000` scan 400s; a cursor-based loop silently stops after one page. Full-bank scans now
**keyset-paginate**: `rank_by=("id","asc")` + an `id > last_seen` filter per page. If you extend the
backend and see a 400 on a range filter, confirm the attribute you keyset on is filterable and
ordered (the primary `id` is).

## HuggingFace 401 on a local model path

Log line:

```
HTTP Request: GET https://huggingface.co/api/models/models/biencoder_v1 "HTTP/1.1 401 Unauthorized"
```

Non-fatal. sentence-transformers sees a relative, `a/b`-shaped path (`models/biencoder_v1`), treats
it as a possible Hub repo id, pings the Hub (→ 401), then loads from the local directory. Silence it
either way:

- Pass an **absolute** model path: `--model-dir "$PWD/models"` (a path that exists is unambiguously
  local, so the Hub is never consulted).
- Or force offline mode: `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

If `/partition` or `/answer` returns `200`, the encoder loaded fine and the `401` is pure noise.
