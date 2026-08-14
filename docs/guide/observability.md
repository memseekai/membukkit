# Observability & telemetry

MEMBUKKIT emits structured spans and metrics through a thin, **dependency-optional** facade over
[Pydantic Logfire](https://logfire.pydantic.dev/) / OpenTelemetry
([`membukkit/telemetry.py`](https://github.com/memseekai/membukkit/blob/main/src/membukkit/telemetry.py)).
Every module calls `from membukkit import telemetry` and uses `telemetry.span(...)`,
`telemetry.counter(...)`, etc. without caring whether telemetry is installed or configured.

Three states, all safe:

| State | Behavior |
|-------|----------|
| `logfire` not installed | Every helper is a no-op. Zero overhead, zero required dependency. |
| Installed but not configured | Spans/metrics are created but nothing is exported. |
| Installed **and** `configure()` has run | Spans/metrics flow to Logfire cloud and/or an OTLP collector. |

## Enabling it

### Install

```bash
pip install "membukkit[observability]"   # or [service], which includes it
```

### In the service (default: on)

`membukkit serve` configures telemetry on startup and auto-instruments FastAPI, the LLM client, and
httpx (see [Memory service](service.md)). Relevant flags:

| Flag | Default | Effect |
|------|---------|--------|
| *(none needed)* | on | Telemetry is configured automatically at startup. |
| `--no-telemetry` | n/a | Disable all instrumentation. |
| `--environment <tag>` | none | Sets the deployment environment tag (e.g. `prod`, `staging`). |
| `--capture-content` | off | Attach **raw** fact/query/LLM text to telemetry. **PII: debug only.** |

### Programmatically

```python
from membukkit import telemetry

telemetry.configure(
    service_name="membukkit",
    environment="prod",        # optional deployment tag
    capture_content=False,     # keep raw text OUT of telemetry (default)
    console=False,             # set True to also print spans to stdout
)
telemetry.instrument_llm()     # OpenAI/Anthropic token usage
telemetry.instrument_httpx()   # covers the Turbopuffer HTTP client
# telemetry.instrument_fastapi(app)  # if you build your own app
```

`configure()` is idempotent and returns `False` (no-op) when `logfire` isn't installed.

### Where the data goes (export auto-detects)

No code change is needed to pick a destination, it's driven by environment variables:

| Env var | Effect |
|---------|--------|
| `LOGFIRE_TOKEN` | Export to Logfire cloud. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Export to any OTLP collector (Grafana, Honeycomb, Jaeger, …). |
| *(both set)* | Export to both. |
| *(neither set)* | Spans/metrics are created but **not sent**: still safe, useful with `console=True` for local inspection. |

Standard `OTEL_*` variables (headers, resource attributes, sampling) are honored by the OpenTelemetry SDK underneath.

## What's auto-instrumented

- **FastAPI**: per-route request spans (headers not captured).
- **LLM clients**, OpenAI and Anthropic calls, with token usage captured automatically. Message
  content is included **only** when `capture_content=True`.
- **httpx**: every HTTP call, which covers the Turbopuffer client (request/response bodies off).
- **System metrics**, CPU/memory/etc. via `instrument_system_metrics()`.

## Metrics

All metric names are prefixed `membukkit.`. Metric **labels** are kept low-cardinality, only
string/bool span attributes become labels; numeric attributes (counts, pool sizes) stay on spans.

| Metric | Type | Unit | Meaning |
|--------|------|------|---------|
| `membukkit.queries` | counter | 1 | Questions answered. |
| `membukkit.facts.ingested` | counter | 1 | New facts written. |
| `membukkit.facts.dedup_skipped` | counter | 1 | Facts skipped as duplicates on upsert. |
| `membukkit.recluster` | counter | 1 | Re-cluster runs. |
| `membukkit.errors` | counter | 1 | Handled errors, tagged by `stage` (`ingest`, `answer`). |
| `membukkit.answer.duration` | histogram | ms | End-to-end `answer()` latency. |
| `membukkit.retrieve.duration` | histogram | ms | Candidate retrieval latency. |
| `membukkit.search.duration` | histogram | ms | Backend search (ANN + BM25 fuse) latency. |
| `membukkit.db.query.duration` | histogram | ms | Turbopuffer server-side query time (from its perf block). |
| `membukkit.embed.duration` | histogram | ms | Query/fact embedding latency (tagged `kind=query\|facts`). |
| `membukkit.rerank.duration` | histogram | ms | Cross-encoder rerank latency. |
| `membukkit.scan_fraction` | histogram | 1 | Fraction of the bank scanned per query. |
| `membukkit.pool_size` | histogram | 1 | Candidate pool size fed to the reranker. |
| `membukkit.tenants.cached` | gauge | 1 | Per-worker count of cached tenant `MemorySystem`s (service only). |

## Spans

| Span | Covers |
|------|--------|
| `memory.ingest` | Distill + store a batch of sessions. |
| `memory.ingest_facts` | Store pre-normalized facts. |
| `memory.supersede` | Knowledge-update patches. |
| `memory.recluster` | Full re-cluster + relabel. |
| `tpuf.search` | One Turbopuffer round trip (attributes: `lanes`, `pool_size`, and the perf block). |

Spans nest under the FastAPI request span in the service, giving a full waterfall per request.

## PII & content scrubbing

By default MEMBUKKIT **never** puts raw fact/query text or LLM message content into telemetry. When
Logfire is configured, these attribute keys are added to its scrubbing patterns:
`memory_text`, `fact_text`, `question`, `answer_text`, `content`.

`capture_content=True` (or `--capture-content`) flips this on, attaching user text and LLM message
bodies. Use it only for local debugging, never in production.

## The `perf` block in the answer trace

Every `AnswerResponse.trace` (see the [HTTP API reference](../reference/http-api.md#trace-object))
carries a `perf` object. Its contents come **directly from Turbopuffer's query `performance`** and
are passed through verbatim (unknown fields are forwarded as-is), so exact keys track the
Turbopuffer SDK. Common fields:

| Field | Meaning |
|-------|---------|
| `server_total_ms` / `query_execution_ms` | Turbopuffer server-side time for the query. |
| `client_total_ms` | Total client-observed time for the round trip. |
| `client_response_ms` | Time to first response byte. |
| `client_body_read_ms` | Time reading the response body. |
| `client_deserialize_ms` | Time deserializing the response. |
| `client_compress_ms` | Client-side request compression time. |
| `cache_hit_ratio` | Fraction of the query served from cache (1.0 = fully warm). |
| `cache_temperature` | `hot` / `warm` / `cold`: where the namespace data was served from. |
| `exhaustive_search_count` | Rows scanned exhaustively (vs. via the ANN index). |
| `approx_namespace_size` | Approximate row count in the namespace. |
| `billable_logical_bytes_queried` / `bytes_queried` | Bytes read for billing/throughput. |
| `last_included_write_at` | Timestamp of the most recent write reflected in this query (consistency). |

The backend also **surfaces** this block onto the `tpuf.search` span and derives the
`membukkit.db.query.duration` metric from `server_total_ms`/`query_execution_ms`, so you get the same
timing whether you read the API trace or your telemetry backend.

> **Reading `perf` to diagnose latency**
>
> - High `server_total_ms` with `cache_temperature: cold` → the namespace was evicted; call
>   `POST /v1/{owner}/warm` on session-open to pre-warm.
> - High `client_total_ms` but low `server_total_ms` → network/serialization, not query cost.
> - Large `exhaustive_search_count` relative to `approx_namespace_size` → the ANN index isn't
>   pruning much; check filters and partition health.