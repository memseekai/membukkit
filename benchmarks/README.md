# Retrieval benchmarks

Document-retrieval benchmarks for MemBukkit. All are **retrieval only**: no
answer generation, no LLM judge, and no API key needed for any default mode.

For the conversational-memory equivalent (session-level Recall@5 on the original
LongMemEval-S split, the metric gbrain and MemPalace publish), see
[`longmemeval_session_recall/`](longmemeval_session_recall/README.md).

```bash
# Shared-corpus multi-hop: the ones that actually separate retrievers
uv run python -m benchmarks.multihop.run --dataset musique --mode chain
uv run python -m benchmarks.multihop.run --dataset 2wiki   --mode dense --limit 200

# Candidate-set retrieval (easier; both systems saturate at k=10)
uv run python -m benchmarks.hotpotqa.run --limit 1000 --seed 42

# Wiring sanity check against QMD's own bundled fixture
uv run python -m benchmarks.qmd.run
```

HotpotQA needs the `bench` extra for its parquet reader
(`uv sync --extra bench`). The multi-hop splits are plain JSON and need nothing
extra.

Three things live here, and they answer different questions:

| | What it tells you |
|---|---|
| [§9 multi-hop](#9-shared-corpus-multi-hop-musique-2wiki-hotpotqa) | how retrieval behaves when the corpus is large and hops are long |
| [§8 head-to-head](#8-head-to-head-against-qmd) | how MemBukkit compares to QMD on identical inputs |
| [`PAPER_RESULTS.md`](PAPER_RESULTS.md) | the research that shaped the retriever, including its negative results |

## Reproducing the comparison yourself

You do not have to trust this harness. MemBukkit reads QMD's fixture format and
emits QMD's report schema, so both systems consume the same two inputs and
produce two diffable JSON files:

```bash
uv run python -m benchmarks.multihop.export_corpus --dataset musique --out /tmp/musique

uv run python -m benchmarks.multihop.run_fixture --corpus /tmp/musique --out mb.json

qmd init && qmd collection add /tmp/musique/docs --name musique && qmd embed
qmd bench /tmp/musique/queries.json --collection musique --json > qmd.json

uv run python -m benchmarks.common.qmd_report membukkit=mb.json qmd=qmd.json
```

The last command prints one table with both systems' backends side by side, and
restricts to queries present in both reports so the comparison stays
like-for-like.

Scoring is QMD's own, ported in
[`qmd_compat.py`](common/qmd_compat.py) from its `score.ts` including the
quirks: `precision_at_k` divides by `min(k, len(expected))`, and unsuffixed
`recall` is computed over the whole result list rather than the top k.
Reproducing QMD's numbers means reproducing its definitions, not correcting
them. Metrics QMD does not define are absent rather than silently added.

MemBukkit's retrieval modes (`dense`, `rerank`, `chain`, `decompose`) occupy the
same `backends` slot QMD fills with `bm25`, `vector`, `hybrid`, `full`, so the
two summary blocks line up column for column. Result, backend, and summary keys
are identical to a real `qmd bench --json` file; the only difference at the top
level is added metadata (`system`, `dataset`, `note`).

One fairness detail that is easy to miss: exported filenames never begin with a
dot. Wikipedia titles such as `.hack//Sign` and `.50 BMG` would otherwise become
dotfiles, which most indexers skip, handing one system a smaller corpus than the
other. See [`common/paths.py`](common/paths.py).

## The retriever under test

Document retrieval is exposed directly, separate from `MemorySystem.search`,
which is built for dated conversational facts rather than documents:

```python
from membukkit.retrieval.rag import Document, RagRetriever

r = RagRetriever(mode="chain")          # dense | rerank | chain | decompose
r.index([Document("d1", "..."), Document("d2", "...")])
hits = r.search("who directed the film?", top_k=5)
```

`chain` is the default and needs no LLM: it reranks to pick the single best
document, harvests that document's entities, appends them to the query, and
ranks everything else against the expanded query. `decompose` adds an LLM loop
on top and accepts either a local model or an API one. See
[`rag.py`](../src/membukkit/retrieval/rag.py).

## 1. What QMD Bench evaluates

[QMD](https://github.com/tobi/qmd) ships a benchmark harness in `src/bench/`.
Its bundled fixture asks 10 queries against 6 markdown documents and checks
whether the expected file appears within a per-query `expected_in_top_k`
(1, 3, or 5). Query types are `exact`, `semantic`, `alias`, `cross-domain`, and
`topical`.

We vendor that fixture verbatim, pinned by commit, under
[`qmd/fixture/`](qmd/fixture) with a `MANIFEST.json` recording the upstream
repo, commit SHA, and a sha256 for every file. Re-fetch with
`python -m benchmarks.qmd.fetch_fixture`.

### Reproducing QMD's scorer

`benchmarks/common/qmd_compat.py` is a direct port of QMD's `score.ts`,
including two things that differ from textbook definitions:

| QMD behaviour | Why it matters |
|---|---|
| `precision_at_k = hits@k / min(k, len(expected))` | With one expected file, a single hit at k=10 scores **1.0**, not 0.1 |
| `recall` is computed over **all** results, not top-k | Answers "found at all", not "found early" |
| `pathsMatch` accepts either path being a suffix of the other | `a/b/x.md` matches `x.md` |

QMD reports recall at 1/3/5 only, and has no nDCG. Our report prints QMD's
numbers under their semantics, plus standard precision@k, recall@10, and
nDCG@10 clearly labelled as extensions.

## 2. Why the bundled QMD fixture is only a sanity test

6 documents and 10 queries is too small and too easy to separate systems.
QMD's own README reports roughly 1.00 for its full pipeline on this fixture,
and MemBukkit also saturates it. Treat it as proof that ingestion, retrieval,
and scoring are wired together correctly, not as evidence about quality.

It is not the basis for any cross-system claim. The head-to-head that is
(§8) runs on HotpotQA, where the task is hard enough to separate systems.

## 3. Why HotpotQA is added

HotpotQA questions require combining **two** documents, so the corpus is bigger
and the task actually discriminates. It also separates two things the single-gold
QMD fixture cannot:

- **Any-support Recall@k** — did we find at least one gold document?
- **All-support Recall@k** — did we find *every* gold document?

With one gold document these are identical. With two they diverge, and the gap
is the interesting signal for multi-document retrieval.

## 4. Dataset and split

- **HotpotQA**, `distractor` config, `validation` split (7,405 questions).
- Source: the Hugging Face parquet conversion of `hotpotqa/hotpot_qa`. The
  official CMU host (`curtis.ml.cmu.edu`) is referenced by the HotpotQA README
  but is frequently unreachable, so it is not used.
- Cached at `~/.cache/membukkit/hotpotqa/`.
- Sampling is deterministic: `(split, limit, seed)` always selects the same
  questions, and selection runs over the id-sorted list so the order of the
  source file cannot change the sample.

### This is candidate-set retrieval, not Wikipedia retrieval

In the distractor setting each question ships its own 10 paragraphs: 2 gold
supporting documents and 8 distractors. Each question is indexed and searched
**in isolation**, so no other question's paragraphs can leak in. These numbers
therefore say nothing about retrieval over full Wikipedia, which is a much
harder task.

## 5. Chunk-level retrieval, document-level scoring

MemBukkit retrieves *facts* (chunks), so one document can occupy several ranks.
All scoring here is document level:

1. Documents are chunked with MemBukkit's own `_chunk_document`, the same
   splitter `membukkit ingest` uses. This matters: the bi-encoder truncates at
   **384 tokens (~1,500 chars)**, and the QMD fixture documents are 1.7–3.4 KB,
   so indexing each as a single unit would embed only its first third.
2. Every chunk keeps its source `doc_id`.
3. Ranked chunks are collapsed to unique documents by **first occurrence**,
   preserving the retriever's ordering (`benchmarks/common/dedup.py`).
4. Retrieval goes **deep** (all chunks) and metrics are computed at cutoffs.
   Asking for only 10 chunks yields just 2–5 distinct documents, which makes a
   document-level Recall@10 impossible to satisfy and understates recall.

### Two configuration facts that materially affect the numbers

**Corpora are ingested undated.** `MemorySystem.search` re-sorts hits
chronologically before returning them, which discards the relevance ranking
that routing and the cross-encoder produced. When every fact is undated,
`datetime_sort_key` returns `-inf` for all of them, the sort is a stable no-op,
and relevance order survives. Give any document a date and rank order silently
becomes date order. `harness.assert_undated` enforces this rather than trusting
it.

**Scan budget is autoscaled, as the product does.** `open_store` calls
`_autoscale_budget`, which forces a full scan below 500 facts, so every real
entry point (CLI, GUI, local HTTP API) full-scans a small store. The harness
applies the same rule. Without it, bucket routing considers ~30% of the corpus
and most candidate documents never reach the ranking at all: measured on 10
HotpotQA questions, only 3–9 of each question's 10 candidates were rankable and
All@10 capped at 0.70 instead of the structurally correct 1.00.

Ingestion is verbatim-only (`distill=False`), so no LLM rewrites document text
before it is scored, and no API calls are made.

## 6. Fairness

- HotpotQA text is used verbatim under a `# {title}` heading; nothing is
  rewritten, reordered, or summarised.
- The answer string is never indexed.
- Supporting-fact labels never appear in document text, document ids, or any
  searchable metadata. They exist only in evaluator-side structures.
- The query is never injected into ingestion.
- Retrieval uses the normal public `MemorySystem.search` path.
- The full retrieval config, encoder, reranker, and MemBukkit version are
  recorded in every results JSON.

## 7. Output

Results are written to `benchmarks/results/{qmd,hotpotqa}_<timestamp>.json`,
containing the config snapshot, aggregate summary, and per-query detail
(query, expected docs, retrieved docs, first relevant rank, latency, chunk
counts). A compact table is printed to stdout.

## 8. Head-to-head against QMD

Both systems were run over the **same 1,000 questions** (seed 42), the same
per-question candidate sets, and the **same metric functions**. QMD indexes the
corpus that `export_corpus.py` writes; its ranked `top_files` are fed through
the identical `benchmarks/common/metrics.py` used for MemBukkit, so any-support
and all-support mean exactly the same thing on both sides.

```bash
python -m benchmarks.hotpotqa.export_corpus --limit 1000 --seed 42 --out /tmp/hotpot_corpus_1k
python -m benchmarks.hotpotqa.run          --limit 1000 --seed 42
python -m benchmarks.hotpotqa.run_qmd      --corpus /tmp/hotpot_corpus_1k --qmd-bin <path>/bin/qmd
```

### Results

**All@5** is the metric that matters here: did retrieval surface *every*
document the question needs? A multi-hop question is unanswerable without both,
so finding one of two is not a partial success, it is a miss.

| System / configuration | Any@1 | All@3 | **All@5** | MRR | nDCG@10 | bridge All@5 | Latency |
|---|---|---|---|---|---|---|---|
| QMD bm25 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 6ms |
| QMD hybrid | 0.744 | 0.632 | 0.834 | 0.859 | 0.858 | 0.794 | 2,401ms |
| QMD full (its recommended path) | 0.744 | 0.710 | 0.860 | 0.863 | 0.871 | 0.819 | 1,457ms |
| **QMD vector (its best)** | **0.932** | 0.724 | 0.861 | **0.958** | 0.923 | 0.823 | 1,222ms |
| MemBukkit, `MemorySystem.search` | 0.872 | 0.623 | 0.766 | 0.923 | 0.888 | 0.705 | 97ms |
| MemBukkit, document-level + entity chain | 0.921 | **0.733** | **0.874** | 0.953 | 0.922 | 0.855 | **47ms** |
| MemBukkit, Qwen3-0.6B encoder + local 1.7B decomposition | 0.921 | **0.773** | **0.908** | 0.954 | **0.926** | **0.884** | seconds |

Reading the table honestly:

- **QMD leads on Any@1 and MRR** (0.932 / 0.958 against 0.921 / 0.953). At
  n=1,000 that gap is under one standard error, so it is a statistical tie
  rather than a QMD win, but it is not a MemBukkit win either.
- **MemBukkit leads on every all-support metric.** +0.047 All@5 and +0.061 on
  bridge questions for the local-decomposition config, which at 3.3σ and 3.4σ
  are well outside sampling noise. The entity-chain config leads by ~1σ, a tie.
- **nDCG@10 is effectively level** across QMD vector (0.923) and both MemBukkit
  document-level configs (0.922, 0.926).
- The pattern is consistent: QMD is marginally better at putting *one* right
  document first; MemBukkit is meaningfully better at retrieving *both*.

### Which MemBukkit configuration is which

This matters, because the three rows are not interchangeable and only one of
them is what `membukkit search` does today.

- **`MemorySystem.search`** is the shipped path. It chunks documents and
  collapses chunks back to documents, which is the right design for dated
  conversational facts with supersession, and a poor fit for 10-document
  candidate-set retrieval. Its 0.766 is the honest number for the shipped
  default on *this* task, and it is below QMD.
- **document-level + entity chain** indexes whole passages, ranks with the
  cross-encoder and RRF, then expands the query with entities drawn from the
  top-ranked document and re-ranks the remainder. No LLM, no new models. It is
  composed from `membukkit.retrieval.multihop` components in the benchmark
  harness and **is not yet a public API**.
- **Qwen3-0.6B encoder + local 1.7B decomposition** changes two things at once,
  so attribute it carefully. It swaps the shipped MPNet bi-encoder for
  `Qwen3-Embedding-0.6B` *and* adds an iterative decomposition loop: split the
  question, answer each sub-question as a short bridge entity, substitute that
  answer into the next, and interleave the rankings. The decomposer is a local
  `qwen3:1.7b`, the same size class as the model QMD uses for query expansion,
  so no API is involved on either side. It costs seconds per query.

  Of the two changes, decomposition does nearly all the work. Holding the
  encoder at Qwen3-0.6B, the entity-chain config scores All@5 0.879 and adding
  decomposition takes it to 0.908. The encoder swap on its own is worth +0.005
  All@5 over the shipped MPNet (0.874 → 0.879), inside noise: **the retrieval
  structure, not the embedder, is what moves these numbers.**

### Caveats

- **QMD's bm25 row is almost certainly an artifact, not a finding.** HotpotQA
  questions are long natural-language sentences and QMD's FTS5 path appears to
  apply implicit AND across terms, so nearly nothing matches; single-keyword
  queries do return results. It should not be read as evidence about QMD's
  lexical search.
- **All@10 is 1.000 for both systems** and is omitted above. With 10 candidates
  per question and deep retrieval, every candidate is ranked by construction.
- **Latency is not measured under matched conditions.** MemBukkit's 47ms and
  QMD's 1,222ms are both single-query means, but the decomposition row ran
  under 8-way concurrency against one local model server and is not directly
  comparable; it is reported as "seconds" rather than a precise figure.
- **This benchmark is candidate-set retrieval**, and an easy instance of it.
  Both systems saturate All@10. Harder multi-hop corpora (MuSiQue, 2Wiki) have
  longer reasoning chains and would separate retrieval strategies further.
- QMD was run at commit `40fb36f`, its `bench` output parsed as-is.

Results JSON for every row is in [`results/`](results).

## 9. Shared-corpus multi-hop: MuSiQue, 2Wiki, HotpotQA

The benchmarks above are *candidate-set* retrieval: each question ships its own
10 paragraphs. That is easy enough that every system saturates All@10 at 1.000,
which makes it useless for separating retrieval strategies. This suite fixes
that by using the **official HippoRAG release splits**, where every question is
answered against the whole corpus.

| Split | Questions | Passages | Gold per question |
|---|---:|---:|---|
| MuSiQue | 1,000 | 11,656 | 2.60 avg (2, 3, and 4-hop) |
| 2WikiMultiHopQA | 1,000 | 6,119 | 2.47 avg |
| HotpotQA | 1,000 | 9,811 | 2.00 |

```bash
uv run python -m benchmarks.multihop.run --dataset musique --mode chain
uv run python -m benchmarks.multihop.run --dataset musique --mode decompose --llm ollama:qwen3:1.7b
```

Splits download once to `~/.cache/membukkit/multihop/`. Sizes are asserted
against the official counts and the loader fails loudly on a mismatch, because
a silently-substituted variant would invalidate every comparison drawn from it.
Each result JSON records the SHA-256 of both the question file and the corpus.

### Why this one discriminates

A 30-question smoke run over MuSiQue with `--mode chain`, scored at the title
level:

| Subset | N | R@2 | R@5 | All@5 | All@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| overall | 30 | 0.486 | 0.678 | 0.333 | 0.500 | 0.939 |
| 2-hop | 17 | 0.588 | 0.735 | 0.471 | 0.706 | 0.931 |
| 3-hop | 12 | 0.361 | 0.611 | 0.167 | 0.250 | 0.944 |
| 4-hop | 1 | 0.250 | 0.500 | 0.000 | 0.000 | 1.000 |

Nothing saturates, and All@k falls away cleanly as the hop count rises. That
gradient is the signal a retrieval benchmark should produce, and the
candidate-set benchmarks cannot show it. (30 questions is a smoke run and is
quoted only to show the shape; it is far too small to report as a result.)

### Comparing another retriever on the same corpus

Use the QMD-protocol path described at the top of this file: one export, two
runners, one comparison command. Gold labels are exported as filenames, which is
what QMD reports and matches on, so both systems are scored against the same
targets.

**No cross-system numbers have been measured on these splits yet**, so nothing
here claims any. A 40-query smoke run of MemBukkit alone, scored under QMD's
scorer over the full 11,656-passage MuSiQue corpus, gives the shape:

| backend | precision | recall | R@1 | R@3 | R@5 | MRR | latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.610 | 0.525 | 0.188 | 0.329 | 0.416 | 0.777 | 28ms |
| rerank | 0.682 | 0.597 | 0.244 | 0.369 | 0.471 | 0.904 | 105ms |
| chain | 0.682 | 0.591 | 0.244 | 0.382 | 0.479 | 0.901 | 207ms |

40 queries is a smoke run, quoted only to show that nothing saturates. It is far
too small to report as a result and no QMD column has been measured to sit
beside it.

### Scoring

Passages carry titles and gold labels are titles, so a ranked passage list is
collapsed to unique titles by first occurrence before any metric is computed,
the same document-level rule the candidate-set benchmarks use. Results are
broken out by hop count where the split provides it.

## 10. What CI checks

The retrieval benchmarks are model- and network-heavy, so CI verifies only what
can be checked without either: metric maths, dataset parsing and its guardrails,
retrieval logic against injected fake models, the vendored QMD fixture
checksums, and that every benchmark entry point imports. Full runs are manual
and their outputs are committed under [`results/`](results).
