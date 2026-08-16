"""Collapse MemBukkit's chunk-level hits into a document-level ranking.

MemBukkit retrieves *facts* (chunks of an ingested document), so a single
document can occupy several consecutive ranks. Scoring that directly would be
wrong in both directions: a document that chunked into ten pieces would crowd
out its competitors, and recall@5 would silently become "recall among however
many documents happened to survive chunking".

The rule here is first-occurrence: walk the ranked hits in order, emit each
document the first time it appears, and drop later chunks of the same document.
That preserves the retriever's own ordering while making the unit of evaluation
a document, which is what both benchmarks score.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence


def collapse_to_documents(
    hits: Sequence,
    doc_id_of: Callable[[object], str] | None = None,
) -> List[str]:
    """Ranked unique document ids, best first, in first-occurrence order.

    ``hits`` is any sequence of retrieval hits in rank order. ``doc_id_of``
    extracts the document id; by default it reads ``doc_id`` and falls back to
    ``doc_name``. Hits with no resolvable document id are skipped rather than
    grouped together under an empty key, since that would merge unrelated
    documents into one phantom result.
    """
    if doc_id_of is None:

        def doc_id_of(h):  # type: ignore[misc]
            return getattr(h, "doc_id", "") or getattr(h, "doc_name", "") or ""

    seen: set[str] = set()
    ordered: List[str] = []
    for hit in hits:
        doc = doc_id_of(hit)
        if not doc or doc in seen:
            continue
        seen.add(doc)
        ordered.append(doc)
    return ordered


def chunk_span_by_document(
    hits: Sequence,
    doc_id_of: Callable[[object], str] | None = None,
) -> Dict[str, int]:
    """How many chunks each document contributed, for diagnostics.

    A large number here means chunking is spreading one document across many
    ranks, which is worth knowing when interpreting precision.
    """
    if doc_id_of is None:

        def doc_id_of(h):  # type: ignore[misc]
            return getattr(h, "doc_id", "") or getattr(h, "doc_name", "") or ""

    counts: Dict[str, int] = {}
    for hit in hits:
        doc = doc_id_of(hit)
        if doc:
            counts[doc] = counts.get(doc, 0) + 1
    return counts
