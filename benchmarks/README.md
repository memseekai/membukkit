# Retrieval benchmarks

Two document-retrieval benchmarks for MemBukkit. Both are **retrieval only**:
no answer generation, no LLM judge, no API keys required.

```bash
uv run python -m benchmarks.qmd.run
uv run python -m benchmarks.hotpotqa.run --limit 100 --seed 42
uv run python -m benchmarks.hotpotqa.run --limit 0            # full split
```

HotpotQA needs the `bench` extra for the parquet reader:
`uv sync --extra bench` (or `pip install "membukkit[bench]"`).

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

**No comparison to QMD has been run.** Nothing here measures QMD itself, so
this directory makes no claim about which system retrieves better.

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

## 8. Running QMD against the same corpus

Not done yet. The HotpotQA corpus generator emits plain markdown
(`# {title}` + paragraph), so it can be written to disk and indexed by QMD to
produce comparable numbers. What is still missing is a flag to dump the
generated corpus per question, and a QMD-side runner. Until both exist and have
actually been run, no cross-system comparison should be stated.
