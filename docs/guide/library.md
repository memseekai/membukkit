# Library API

For agents, start with the thin facade:

```python
from membukkit import Memory

mem = Memory.from_pretrained(llm="openai:gpt-4o-mini")
mem.add("Rent is $2100.", subject="alex", date="2024-01-10")
r = mem.ask("How much is rent?", as_of="2024-06-01")
# r.answer, r.scan_fraction, r.evidence[*].status  (current | superseded)
```

`add` returns a `WriteReport` (`n_stored`, `superseded`, `warnings`, `status`). Local agents can
also call `POST /api/v1/{store}/add|search|ask` on `membukkit ui`.

The power object is [`MemorySystem`](../code-walkthrough.md), sessions, prompt packs, backends.

## Constructing a system

```python
from membukkit import MemorySystem, RetrievalConfig, ModelConfig, PromptConfig

mem = MemorySystem.from_pretrained(
    models=ModelConfig(model_dir="./models"),
    retrieval=RetrievalConfig(num_buckets=24, scan_budget=0.3, select="hybrid", top_k=10),
    llm="openai:gpt-4o-mini",
    prompts=PromptConfig.default(),
)
```

## Ingesting data

MEMBUKKIT works on **any** source you can turn into dated text turns, chat logs, a calendar,
emails, support tickets, meeting notes. The contract: `ingest()` takes a list of **sessions**, where
each session is a list of `{"role", "content"}` turns, plus an optional date per session.

```python
mem.ingest(
    sessions=[
        [{"role": "me",     "content": "Booked the dentist for next Tuesday at 10."},
         {"role": "partner","content": "Great, I'll drop the kids at school then."}],
        [{"role": "me",     "content": "Started learning Portuguese on Duolingo today."}],
    ],
    dates=["2024/06/03", "2024/06/07"],
    subject="me",   # optional: write facts in the named person's voice
)
```

### Structured sources (calendar, rows)

Flatten each event/row into a turn and give it the event date:

```python
events = [
    {"date": "2024/05/12", "text": "Flight to Lisbon, TAP 1038, seat 14C"},
    {"date": "2024/05/20", "text": "Quarterly review with Dana, 3pm, Room 4"},
]
mem.ingest(
    sessions=[[{"role": "user", "content": e["text"]}] for e in events],
    dates=[e["date"] for e in events],
)
```

### Pre-normalized facts (skip the distiller)

If your facts are already atomic and dated, write them **directly** with `ingest_facts`, it skips
the LLM distiller entirely. This is useful for calendar events, CRM records, or any pre-structured data.

```python
mem.ingest_facts(facts, subject="me")   # direct, no LLM
```

> **Dates drive temporal reasoning**
>
> Use `YYYY/MM/DD` (or `YYYY/MM/DD HH:MM`), a `datetime`, or an ISO-8601 string. Missing dates are
> allowed but temporal questions degrade. `subject="Name"` attributes facts to that person and
> avoids mixing up other people's details.

## Answering

```python
res = mem.answer("When is my dentist appointment?", question_date="2024/06/10")
print(res.answer)
print(res.trace.scan_fraction, res.trace.opened_buckets)
print(res.facts)   # chronologically-ordered top-k fact lines used to answer
```

`AnswerResult` carries `answer`, the supporting `facts`, and a `RetrievalTrace` (`opened_buckets`,
`scan_fraction`, `n_facts`, `n_scanned`, `reader_type`, `backend`, …).

## Inspecting the buckets (explainability)

```python
part = mem.partition()
print(part["k_eff"], "buckets")

# Auto-name buckets with the LLM (Work/Manager, Diet/Health, Travel, ...)
labels = mem.label_buckets()
print(labels)   # {0: "Diet & health", 1: "Work & scheduling", ...}
```

## Configuration

Swap models, edit prompts, and tune retrieval without touching code:

```python
# Different encoder / reranker, different LLM
mem = MemorySystem.from_pretrained(
    models=ModelConfig(encoder="my-biencoder", reranker="my-reranker/model"),
    llm="anthropic:claude-sonnet-4-20250514",
)

# Steer extraction / reading (Mem0-style overlays — see Customization)
prompts = PromptConfig(
    extraction_instructions="Extract ONLY order ids and ship dates. Skip chitchat.",
    reader_instructions="Prefer the most recent dated order fact.",
)
# Or load a shipped pack: from membukkit.prompts import load_prompt_pack
# prompts = load_prompt_pack("customer_support")

mem = MemorySystem.from_pretrained(prompts=prompts)

# Tune retrieval behavior
cfg = RetrievalConfig(
    num_buckets=12,           # fewer buckets for smaller banks
    scan_budget=0.5,          # scan more for higher recall
    select="cosine",          # "hybrid" (default) | "cosine" | "xenc"
    top_k=20,
    bucket_mode="multiaxis",  # add entity + time axes alongside topic
)
```

Full prompt fields, packs, and the GUI editor: **[Customization](customization.md)**.

### `RetrievalConfig` knobs

| Field | Meaning |
|-------|---------|
| `num_buckets` | Target KMeans clusters (K). More buckets suit larger banks. |
| `scan_budget` | Fraction of memory to scan before stopping bucket-opening (default 0.3). |
| `scan_budget_reason` / `scan_budget_temporal` | Deeper budgets for reasoning / temporal queries. |
| `select` | Within-region ranking: `hybrid` (RRF of cosine + cross-encoder), `cosine`, or `xenc`. |
| `top_k` / `reasoning_top_k` | Facts returned for normal vs. multi-fact reasoning queries. |
| `bucket_mode` | `topic` (default) or `multiaxis` (topic + entity + time). |
| `retrieval_mode` | `gated` (bucket-gated scan) or `open` (ANN over the whole bank). See [Storage backends](storage.md#gated-vs-open-retrieval). |
| `lexical_lane` | Opt-in BM25 lane (default `False`). See below. |
| `lexical_top_k` | BM25 hits considered per lane before the union (default 20). |

### Lexical lane (optional BM25) {#lexical-lane}

Retrieval is dense by default: the router opens topic buckets by embedding similarity, and
that is the path every published number was measured with. Embeddings are weakest on exact
strings, though. An error code, a filename, a ticket id, or a rare surname can sit in a
bucket the router never opens, and no amount of reranking recovers a fact that was never
admitted.

Turning on the lexical lane adds a second way in:

```bash
pip install "membukkit[bm25]"
```

```python
from membukkit import Memory
from membukkit.config import RetrievalConfig

mem = Memory.from_pretrained(retrieval=RetrievalConfig(lexical_lane=True))
```

What changes when it is on:

- **BM25 runs over the whole bank** (per lane, so verbatim and atomic are searched
  separately under `union`), and its top hits are **added** to the routed pool. Routed
  candidates are never dropped or reordered by admission.
- **Ranking gains a third signal.** With `select="hybrid"` the BM25 score joins cosine and
  the cross-encoder in the RRF fusion. With `select="cosine"` the lane still admits
  candidates but does not vote on order.
- **The receipt gains two fields.** `lexical_added` counts candidates the lane contributed
  that routing had missed, and `lexical_scanned` reports how many facts the lexical pass
  covered. Read them alongside `scan_frac`, which continues to mean what it always did:
  the fraction of the bank *dense routing* opened. A lexical pass touches every row by
  construction, so the two numbers answer different questions.
- **Cost.** The pool grows past `rerank_cap`, so the cross-encoder scores more candidates
  per query. The index itself is built once per lane and rebuilt when the fact count
  changes; for typical local stores that is well under a second.

Matching is deliberately plain: lowercased alphanumeric tokens with `snake_case` kept
intact, no stemming and no stopword list, so a query term matches the stored term or it
does not. A document sharing no query term is not a hit at all.

> **Not the Turbopuffer `bm25_lane`**
>
> `RetrievalConfig.bm25_lane` is a different, older switch: it controls the server-side
> lexical lane on the [Turbopuffer backend](storage.md) and has no effect on local
> stores. `lexical_lane` is the local one.

See the full pipeline in **[Method](../METHOD.md)** and the annotated code paths in the
**[Code walkthrough](../code-walkthrough.md)**.
