# Memory service

`membukkit.service` is a multi-tenant FastAPI app over the Turbopuffer backend. Expensive models
(bi-encoder + cross-encoder) load **once** and are shared across tenants; each tenant gets a
lightweight `MemorySystem` bound to its own namespace and cached per worker (LRU).

## Install & run

```bash
pip install "membukkit[service]"

export TURBOPUFFER_API_KEY=...
export TURBOPUFFER_REGION=...
export OPENAI_API_KEY=...          # the reader LLM used by /answer

membukkit serve \
  --model-dir "$PWD/models" \
  --llm openai:gpt-4o-mini \
  --retrieval-mode gated \
  --port 8080
```

Or point uvicorn at the env-configured app directly:

```bash
uvicorn membukkit.service.app:app --host 0.0.0.0 --port 8080
```

> **Use an absolute `--model-dir`**
>
> A relative path like `models/biencoder_v1` makes sentence-transformers ping the Hugging Face
> Hub (→ a harmless `401`) before falling back to local weights. An **absolute** path is
> recognized as local and skips the lookup. See
> [Troubleshooting](troubleshooting.md#huggingface-401-on-a-local-model-path).

### `serve` flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--host` / `--port` | `0.0.0.0` / `8080` | Bind address. |
| `--model-dir` | *(none)* | Directory holding encoder/reranker weights. |
| `--encoder` / `--reranker` | `biencoder_v1` / `reranker_v2/model` | Weight subpaths / model IDs. |
| `--llm` | `openai:gpt-4o-mini` | Reader + distiller backend. |
| `--region` | env `TURBOPUFFER_REGION` | Turbopuffer region. |
| `--namespace-prefix` | `mem_` | Owner→namespace prefix. `""` = use owner id verbatim. See below. |
| `--retrieval-mode` | `gated` | `gated` or `open` (see [Storage backends](storage.md#gated-vs-open-retrieval)). |
| `--vector-dtype` | `f16` | `f16` or `f32`. |
| `--environment` / `--capture-content` / `--no-telemetry` | n/a | Observability (Logfire/OTel). |

## Owner → namespace mapping

Every route is scoped to an `{owner}` path segment, which maps 1:1 to a Turbopuffer namespace as
`<prefix><owner>` (after sanitizing the owner to `[A-Za-z0-9_-]`). The prefix is configurable via
`--namespace-prefix` / `ServiceConfig.namespace_prefix` (default `mem_`).

```python
from membukkit.service.manager import namespace_for

namespace_for("acme")                 # -> "mem_acme"      (default multi-tenant prefix)
namespace_for("acme", prefix="")      # -> "acme"          (address a namespace verbatim)
```

!!! example "Pointing the service at a pre-existing namespace"
    To serve a bespoke namespace such as `acme-team-1` directly (rather than `mem_acme-team-1`),
    start the server with an **empty** prefix and use the namespace name as the owner:

    ```bash
    membukkit serve --model-dir "$PWD/models" --namespace-prefix "" --retrieval-mode open --port 8080
    curl -s localhost:8080/v1/acme-team-1/partition | python3 -m json.tool
    ```

    With an empty prefix the owner id *is* the namespace, so all tenants collapse onto whatever
    their owner id spells, fine for single-tenant testing, but keep `mem_` for real multi-tenancy.

## HTTP API

The full endpoint spec, request/response models, status codes, and per-route examples, is in the
**[HTTP API reference](../reference/http-api.md)**. The running server also serves interactive docs
at **`/docs`** (Swagger UI), **`/redoc`**, and the raw schema at **`/openapi.json`**. Summary:

| Method & path | Body / effect |
|---------------|---------------|
| `GET /health` | Liveness check. |
| `POST /v1/{owner}/ingest` | `{sessions, dates?, subject?}`: distill + store conversation turns. |
| `POST /v1/{owner}/answer` | `{question, question_date?, identity?}` → `{answer, facts, trace}`. |
| `GET /v1/{owner}/partition` | `{k_eff, version, sizes}`: builds the partition on first call. |
| `POST /v1/{owner}/label_buckets` | LLM-labeled bucket names. |
| `POST /v1/{owner}/warm` | Prefetch centroids / warm the namespace cache. |
| `POST /v1/{owner}/recluster` | Re-cluster **if** growth warrants it (`maybe_recluster`). |
| `DELETE /v1/{owner}` | Delete the tenant's namespace. |

### Examples

```bash
# health
curl -s localhost:8080/health

# confirm a namespace is loaded / partitioned
curl -s localhost:8080/v1/acme/partition | python3 -m json.tool

# ask a question
curl -s -X POST localhost:8080/v1/acme/answer \
  -H 'content-type: application/json' \
  -d '{"question":"What is scheduled in Barcelona?","question_date":"2026-07-01"}' \
  | python3 -m json.tool
```

The `/answer` response includes the retrieval `trace` (`opened_buckets`, `scan_fraction`,
`n_scanned`, `reader_type`, `backend`, `perf`).

> **`/ingest` is for conversation sessions**
>
> The service `/ingest` endpoint takes **sessions** (which go through the distiller). To load
> pre-normalized facts (e.g. calendar events), use `MemorySystem.ingest_facts` in-process, the
> service does not expose an endpoint for pre-normalized facts.

## Observability

With `membukkit[observability]` (or `[service]`), the app auto-instruments FastAPI, the LLM client,
and httpx (covering Turbopuffer's HTTP calls) via Logfire/OpenTelemetry. Disable with
`--no-telemetry`. `--capture-content` additionally logs raw fact/query/LLM text (PII, debug only).

See **[Observability & telemetry](observability.md)** for the full metric/span catalog, export
configuration (`LOGFIRE_TOKEN` / `OTEL_EXPORTER_OTLP_ENDPOINT`), and how to read the `perf` block.
