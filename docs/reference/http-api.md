# HTTP API reference

Complete reference for the MEMBUKKIT memory service (`membukkit.service.app`). For how to run it and
configure the owner→namespace mapping, see the [Memory service guide](../guide/service.md).

> **Two HTTP surfaces, this page covers the multi-tenant one**
>
> - **Multi-tenant service** (`membukkit serve`, default port **8080**, Turbopuffer-backed):
>   `/v1/{owner}/…`, **documented on this page**.
> - **Local GUI server** (`membukkit ui`, default port **8377**, disk stores): a separate
>   single-user API, `POST /api/v1/{store}/add|search|ask` plus `GET /api/health`, with
>   curl examples in [Agents → local HTTP](../guide/agents.md#local-http).

- **App title / version**: `MEMBUKKIT Memory Service` `1.0`
- **Base path**: all tenant routes are under `/v1/{owner}`
- **Content type**: `application/json` for every request/response body
- **Auth**: **none**, the service does not authenticate requests. Tenancy is client-asserted via
  the `{owner}` path segment. Deploy only on a trusted network, or put an authenticating gateway in
  front. See [Design note: tenant identity](#design-note-tenant-identity).

## Interactive / machine-readable docs

FastAPI serves live, auto-generated API docs from the running server:

| URL | What |
|-----|------|
| `/docs` | Swagger UI: try every endpoint in the browser. |
| `/redoc` | ReDoc rendering of the same spec. |
| `/openapi.json` | The raw OpenAPI 3 schema (for client codegen, Postman, etc.). |

```bash
open http://localhost:8080/docs
curl -s localhost:8080/openapi.json | python3 -m json.tool | head
```

## The `{owner}` path parameter

Every `/v1/...` route is scoped to an `{owner}`. The service maps it to a Turbopuffer namespace as
`<namespace_prefix><owner>` (default prefix `mem_`), after sanitizing the owner to
`[A-Za-z0-9_-]` (max 48 chars). Set `--namespace-prefix ""` to address a namespace verbatim. See
[owner → namespace mapping](../guide/service.md#owner-namespace-mapping).

---

## Data models

### `Turn`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `role` | string | `"user"` | Speaker label (e.g. `user`, `assistant`, or any name). |
| `content` | string | `""` | The turn text. |

### `IngestRequest`

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `sessions` | array of arrays of `Turn` | yes | One inner array per conversation/session. |
| `dates` | array | no | One date per session (`datetime`, ISO-8601, or `YYYY/MM/DD`). Parsed server-side. |
| `subject` | string | no | Attribute distilled facts to this person. |

### `AnswerRequest`

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `question` | string | yes | The natural-language question. |
| `question_date` | string / datetime | no | Reference "now" for temporal reasoning. Parsed server-side. |
| `identity` | string | no | Who is asking (used by the reader for first-person framing). |

### Date/time values

The `dates[]` (ingest) and `question_date` (answer) parameters are parsed server-side by
`parse_datetime`. Accepted forms:

| Form | Examples |
|------|----------|
| ISO-8601 (canonical) | `2026-07-01`, `2026-07-01T09:30:00`, `2026-07-01T09:30:00+02:00`, `2026-07-01T09:30:00Z` (trailing `Z` accepted) |
| Legacy slash/dash | `2026/07/01`, `2026/07/01 09:30`, `2026/07/01 (Wed) 09:30`, `2026-07-01 09:30` |
| Absent / unparseable | `null`, `""`, or any unrecognized string → treated as **no date** |

Timezone-aware inputs stay aware; naive inputs stay naive. A bare date becomes midnight of that day.

### `AnswerResponse`

| Field | Type | Notes |
|-------|------|-------|
| `answer` | string | The generated answer. |
| `facts` | array of string | Chronologically-ordered top-k fact lines used to answer. |
| `trace` | object | Retrieval trace: see below. |

### `trace` object

| Field | Type | Meaning |
|-------|------|---------|
| `opened_buckets` | array | Buckets that fired (or post-hoc buckets in open mode) with hits/size. |
| `scan_fraction` | float | Fraction of the bank scanned for this query. |
| `n_facts` | int | Total live facts in the bank. |
| `n_scanned` | int | Candidates the reranker considered. |
| `k_total` | int | Number of topic buckets (`k_eff`). |
| `reader_type` | string | `dated` \| `reasoning` \| `recommendation`. |
| `ranked_fact_times` | array | Timestamps of the ranked facts. |
| `backend` | string | `turbopuffer` \| `memory`. |
| `perf` | object | Backend timing/telemetry counters: passed through from Turbopuffer. Field-by-field in [Observability → the `perf` block](../guide/observability.md#the-perf-block-in-the-answer-trace). |

---

## Endpoints

### `GET /health`

Liveness probe. No auth, no owner.

```bash
curl -s localhost:8080/health
# {"status":"ok"}
```

**200** → `{"status": "ok"}`

---

### `POST /v1/{owner}/ingest`

Distill conversation sessions and store the resulting facts.

**Request body parameters** (JSON):

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `sessions` | array of arrays of [`Turn`](#turn) | **yes** | n/a | One inner array per conversation/session. |
| `sessions[][].role` | string | no | `"user"` | Speaker label for the turn (`user`, `assistant`, or any name). |
| `sessions[][].content` | string | no | `""` | The turn text. |
| `dates` | array of [date/time](#datetime-values) \| `null` | no | `null` | One date per session, **aligned by index** (`dates[i]` → `sessions[i]`). Provide the same length as `sessions`, or omit for undated. Each element is parsed server-side. |
| `subject` | string \| `null` | no | `null` | Attribute the distilled facts to this person (first-person voice; avoids mixing up other people's details). |

```bash
curl -s -X POST localhost:8080/v1/acme/ingest \
  -H 'content-type: application/json' \
  -d '{
        "sessions": [
          [{"role":"user","content":"I switched to a vegan diet last month."}],
          [{"role":"user","content":"My manager is Dana; we have morning standups."}]
        ],
        "dates": ["2024/06/01", "2024/06/10"],
        "subject": "user"
      }'
```

- **200** → `{"ok": true, "n_facts": <int>}` (`n_facts` = total facts in the namespace after ingest)
- **422** → request body failed validation
- **500** → `{"detail": "<error>"}` (distillation/storage failure; increments `membukkit.errors{stage=ingest}`)

> **Sessions only**
>
> This endpoint runs the LLM distiller. To load **pre-normalized** facts (e.g. calendar events),
> use `MemorySystem.ingest_facts` in-process; the service exposes no endpoint for them.

---

### `POST /v1/{owner}/answer`

Answer a question from the owner's memory. **Response**: [`AnswerResponse`](#answerresponse).

**Request body parameters** (JSON):

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `question` | string | **yes** | n/a | The natural-language question. |
| `question_date` | [date/time](#datetime-values) \| `null` | no | `null` | Reference "now" for temporal reasoning (e.g. "last month"). Parsed server-side. |
| `identity` | string | no | `""` | Who is asking: the reader uses it for first-person framing. |

```bash
curl -s -X POST localhost:8080/v1/acme/answer \
  -H 'content-type: application/json' \
  -d '{"question":"What diet is the user on now?","question_date":"2024-07-01"}' \
  | python3 -m json.tool
```

- **200** → `AnswerResponse` (`answer`, `facts`, `trace`)
- **422** → invalid body (e.g. missing `question`)
- **500** → `{"detail": "<error>"}` (increments `membukkit.errors{stage=answer}`)

---

### `GET /v1/{owner}/partition`

Return (building on first call) the topic partition.

```bash
curl -s localhost:8080/v1/acme/partition | python3 -m json.tool
```

**200** → `{"k_eff": <int>, "version": <int>, "sizes": {"<bucket>": <count>, ...}}`

> **Empty result?**
>
> `{"k_eff": 0, "sizes": {}}` means no fact vectors were found to cluster, usually a wrong
> namespace/prefix, a region mismatch, or an empty bank. See
> [Troubleshooting](../guide/troubleshooting.md#partition-returns-k_eff-0-sizes).

---

### `POST /v1/{owner}/label_buckets`

Ask the LLM to name each topic bucket from its exemplar facts. **No request body.**

```bash
curl -s -X POST localhost:8080/v1/acme/label_buckets | python3 -m json.tool
```

**200** → `{"<bucket>": "<label>", ...}` (e.g. `{"0": "Diet & health", "1": "Work & scheduling"}`).
Returns `{}` if there is no usable partition yet.

---

### `POST /v1/{owner}/warm`

Best-effort prefetch of the partition centroids and namespace cache (call on session-open so the
first real query doesn't pay the cold fetch). Never errors on a cold/missing namespace.
**No request body.**

```bash
curl -s -X POST localhost:8080/v1/acme/warm
# {"ok": true}
```

**200** → `{"ok": true}`

---

### `POST /v1/{owner}/recluster`

Re-cluster **only if** the bank has grown past the configured threshold (`maybe_recluster`).
**No request body.**

```bash
curl -s -X POST localhost:8080/v1/acme/recluster
# {"ok": true, "reclustered": false}
```

**200** → `{"ok": true, "reclustered": <bool>}` (`reclustered` is `true` only if a re-cluster ran)

> **Forcing a full relabel**
>
> This endpoint no-ops when the bank hasn't grown. To force a full centroid rebuild + relabel of
> every fact (e.g. after a bulk `ingest_facts` load), call `backend.recluster()` in-process, see
> [Storage backends → Re-clustering](../guide/storage.md#re-clustering).

---

### `DELETE /v1/{owner}`

Delete the tenant's entire namespace and drop it from the worker cache. **Irreversible.**

```bash
curl -s -X DELETE localhost:8080/v1/acme
# {"ok": true}
```

**200** → `{"ok": true}`

---

## Status codes at a glance

| Code | When |
|------|------|
| `200` | Success. |
| `422` | Request body failed pydantic validation (`ingest`, `answer`). |
| `500` | Unhandled server/backend error. `ingest` and `answer` return `{"detail": ...}`; other routes surface FastAPI's default error. |

## Errors & observability

`ingest` and `answer` catch exceptions, log them, increment the `membukkit.errors` counter (tagged by
`stage`), and return `500 {"detail": "<message>"}`. When telemetry is enabled (default; disable with
`--no-telemetry`), FastAPI, the LLM client, and httpx (covering Turbopuffer) are auto-instrumented
via Logfire/OpenTelemetry.

## Design note: tenant identity

The service is **unauthenticated** and derives the tenant from the client-supplied `{owner}` path
segment, appropriate for a trusted/internal network only. For a public deployment, front it with an
authenticating gateway (or add an auth dependency) and resolve the tenant from the authenticated
principal rather than trusting the path. The owner-in-path design itself is deliberate, it keeps
`GET`/`DELETE` consistent with `POST`, and exposes the tenant to edge routing, rate limiting, and
tracing. Prefer **opaque owner ids** (a UUID, not an email) to avoid leaking PII into access logs.
