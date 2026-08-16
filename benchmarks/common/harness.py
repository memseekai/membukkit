"""Shared MemBukkit retrieval harness for the benchmarks.

Two things in here matter more than the plumbing.

**Undated ingestion is mandatory.** ``MemorySystem.search`` re-sorts hits into
chronological order before returning them (``_order_temporal`` in
``membukkit/pipeline.py``), which discards the relevance ranking that routing
and the cross-encoder produced. When every fact is undated, ``datetime_sort_key``
returns ``-inf`` for all of them, Python's stable sort leaves the order alone,
and the relevance ranking survives to the caller. Give any document a date and
rank order silently becomes date order, which would make every rank metric here
meaningless. :func:`assert_undated` enforces that rather than trusting it.

**Verbatim-only ingestion.** Documents are indexed as written, with no LLM
distillation step rewriting them before scoring. That keeps this a retrieval
benchmark, costs no API calls, and matches how a document search engine indexes
a corpus.

The encoder is the expensive thing to construct, so build one system and call
:func:`reset` between corpora instead of rebuilding per query.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass
class Document:
    """One indexable document with a stable id."""

    doc_id: str
    text: str
    title: str = ""


def build_retrieval_system(llm: str = "openai:gpt-4o-mini"):
    """A verbatim-only MemorySystem. No distiller, so no LLM calls on ingest."""
    from membukkit.cli.common import build_system

    return build_system(llm=llm, distill=False)


def reset(mem) -> None:
    """Drop every fact so the next corpus is searched in isolation."""
    mem.backend.clear()


def chunk_document(text: str) -> List[List[Dict[str, str]]]:
    """Split text with MemBukkit's own document chunker.

    Uses the same paragraph splitter ``membukkit ingest`` applies to a markdown
    file, so the benchmark indexes documents the way the product does. This
    matters: the bi-encoder truncates at 384 tokens (~1500 chars), and the
    fixture documents are 1.7-3.4 KB, so indexing a whole document as one unit
    would embed roughly its first third and silently drop the rest.
    """
    from membukkit.ingest.parsers import _chunk_document

    # _chunk_document returns (sessions, dates); we ingest undated, so drop dates.
    sessions, _dates = _chunk_document(text)
    return sessions or [[{"role": "user", "content": text}]]


def ingest_documents(mem, docs: Sequence[Document]) -> int:
    """Ingest documents undated, chunked as MemBukkit would, keeping the doc id.

    Applies the same scan-budget autoscaling the CLI applies when it opens a
    store. Without it the benchmark would run at the default 0.3 budget, so
    bucket routing would consider only ~30% of a small corpus and most
    candidate documents would never reach the ranking at all. Every real entry
    point (``membukkit search``, the GUI, the local HTTP API) goes through
    ``open_store`` and therefore autoscales, so skipping it would measure a
    configuration no user runs, and would understate recall.
    """
    from membukkit.cli.common import _autoscale_budget

    for doc in docs:
        sessions = chunk_document(doc.text)
        mem.ingest(
            sessions=sessions,
            dates=None,  # undated on purpose; see module docstring
            doc_id=doc.doc_id,
            doc_name=doc.title or doc.doc_id,
            doc_type="document",
        )
    _autoscale_budget(mem)
    return mem.backend.count()


def assert_undated(mem) -> None:
    """Fail loudly if anything carries a timestamp.

    A single dated fact flips ``search`` into chronological order, so this is
    the difference between measuring retrieval and measuring calendar order.
    """
    times = [t for t in getattr(mem.backend, "_times", []) if t is not None]
    if times:
        raise AssertionError(
            f"{len(times)} ingested fact(s) carry a timestamp. MemBukkit sorts search "
            "hits chronologically, so ranks would reflect dates rather than relevance. "
            "Ingest benchmark corpora undated."
        )


def search_documents(
    mem, query: str, chunk_k: int | None = None
) -> Tuple[List[str], float, List]:
    """Run one query and return (ranked doc ids, latency ms, raw chunk hits).

    ``chunk_k`` is how many *chunks* to retrieve before collapsing to documents,
    and it must be generous. A document occupies as many ranks as it has
    matching chunks, so asking for 10 chunks typically yields only 2-5 distinct
    documents, which makes a document-level Recall@10 impossible to satisfy and
    silently understates recall for queries with several gold documents.

    Defaulting to the whole corpus retrieves deep and evaluates at cutoffs,
    which is the standard IR approach and gives the complete document ranking.
    These corpora are small (hundreds of chunks), so the cost is negligible.
    """
    from benchmarks.common.dedup import collapse_to_documents

    if chunk_k is None:
        chunk_k = max(int(mem.backend.count()), 1)

    start = time.perf_counter()
    result = mem.search(query, top_k=chunk_k)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return collapse_to_documents(result.hits), latency_ms, list(result.hits)


def config_snapshot(mem) -> Dict:
    """Record the retrieval configuration the numbers were produced under."""
    from dataclasses import asdict

    import membukkit

    cfg = asdict(mem._retrieval)
    encoder = getattr(mem, "_encoder", None)
    reranker = getattr(mem, "_reranker", None)
    return {
        "membukkit_version": getattr(membukkit, "__version__", "unknown"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "retrieval": cfg,
        "encoder": getattr(encoder, "model_name", None) or type(encoder).__name__,
        "reranker": (type(reranker).__name__ if reranker is not None else None),
        "distiller": None,
        "ingestion": "verbatim-only, undated",
        "note": (
            "Ranks come from MemorySystem.search on an undated corpus, where the "
            "chronological presentation sort is a stable no-op and relevance order "
            "is preserved."
        ),
    }
