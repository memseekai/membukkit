"""Storage backend abstraction for MEMBUKKIT memory.

`MemorySystem` no longer owns the fact arrays directly; it talks to a
`MemoryBackend`. Two implementations ship:

  - `InMemoryBackend`  — numpy cosine + KMeans, byte-for-byte the original
                         behaviour (eval/tests run on this, unchanged).
  - `TurbopufferBackend` — persistent, multi-tenant, namespace-per-owner.

The seam is deliberately narrow: the backend owns fact storage, embedding of
new facts, and *candidate generation* (routing + the explainability trace). The
cross-encoder rerank and the final RRF fusion stay in `MemorySystem`, because
the reranker is a model the system owns (and, in the service, a remote GPU
client).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


def content_id(
    text: str,
    subject: Optional[str] = None,
    date: Optional[datetime] = None,
    kind: str = "atomic",
) -> str:
    """Stable content-hash id for a fact.

    Idempotent upsert: the same fact distilled twice from the same (dated)
    session maps to one row. Scoped by `subject` so two personas can share a
    namespace without colliding (namespace isolation is the primary boundary;
    this is defence-in-depth), and by the fact's calendar `date` so a recurring
    fact stated again in a later session is KEPT as a new dated observation
    rather than silently dropped (re-ingesting the same session still dedups —
    it carries the same date). Undated facts dedup on content alone.

    `kind` (verbatim vs atomic) is folded in so the two retrieval lanes of the
    union never collide on an id when the same text appears in both — which also
    keeps `search()` citation refs unique. Atomic keeps the historical key shape
    (no kind prefix) so ids in existing atomic-only namespaces are unchanged.
    """
    norm = " ".join((text or "").lower().split())
    date_key = date.date().isoformat() if date is not None else ""
    if kind == "atomic":
        key = f"{subject or ''}\x1f{date_key}\x1f{norm}"
    else:
        key = f"{kind}\x1f{subject or ''}\x1f{date_key}\x1f{norm}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:32]


@dataclass
class FactRecord:
    """A fact as handed to the backend for storage.

    `vector` is optional: the backend embeds facts that arrive without one
    (so only *new* facts are encoded — see `MemoryBackend.upsert_facts`).
    """

    text: str
    timestamp: Optional[datetime] = None
    tag: str = "NEW_OBS"
    source_session: Optional[str] = None
    source_speaker: Optional[str] = None
    subject: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    time_bucket: str = "unknown"
    kind: str = "atomic"  # "atomic" (distilled/structured) | "verbatim" (raw turn)
    vector: Optional[np.ndarray] = None
    id: str = ""
    # --- document provenance (optional) ---
    # `doc_id`/`doc_name` identify the source document/file the fact came from;
    # `source_ref` points inside it (e.g. "session:3/turn:5" or "chunk:2") so
    # UIs can drill from a fact back to the exact source passage.
    doc_id: str = ""
    doc_name: str = ""
    source_ref: str = ""
    # Supersession (knowledge updates): older fact points at the newer id.
    superseded_by: str = ""
    valid_to: Optional[datetime] = None

    def ensure_id(self) -> str:
        if not self.id:
            self.id = content_id(self.text, self.subject, self.timestamp, self.kind)
        return self.id


@dataclass
class Candidate:
    """A retrieval candidate handed back to the reranker.

    Decouples the pipeline from list indices: everything the cross-encoder and
    the temporal presenter need travels on the candidate itself.
    """

    text: str
    timestamp: Optional[datetime] = None
    cosine: float = 0.0
    # BM25 score when the optional lexical lane is on; 0.0 otherwise (and for
    # candidates the lane did not score). See CandidatePool.has_lexical.
    lexical: float = 0.0
    topic_bucket: int = -1
    entities: List[str] = field(default_factory=list)
    time_bucket: str = "unknown"
    kind: str = ""
    id: str = ""
    doc_id: str = ""
    doc_name: str = ""
    source_ref: str = ""
    superseded_by: str = ""
    valid_to: Optional[datetime] = None


@dataclass
class CandidatePool:
    """Result of `MemoryBackend.candidates`: the pool + the routing trace.

    `has_cosine` tells `MemorySystem` whether per-candidate cosine scores are
    meaningful for RRF fusion (true only when the routing path computed them,
    matching the original `fuse_cos` semantics).
    """

    candidates: List[Candidate] = field(default_factory=list)
    trace: Dict = field(default_factory=dict)
    has_cosine: bool = False
    # True when the optional BM25 lane ran and `Candidate.lexical` carries
    # meaningful scores, so fusion can use it as a third ranking signal.
    has_lexical: bool = False


@runtime_checkable
class MemoryBackend(Protocol):
    """Storage + candidate-generation contract."""

    def clear(self) -> None:
        """Drop all in-process state for this bank (used to reset between eval instances)."""
        ...

    def upsert_facts(self, facts: Sequence[FactRecord], on_progress=None) -> int:
        """Store facts idempotently (dedup on id). Embeds any without a vector.

        Returns the number of *new* facts written.
        ``on_progress`` is an optional ``Callable[[ProgressEvent], None]``.
        """
        ...

    def count(self) -> int:
        """Number of live facts in the bank."""
        ...

    def count_kind(self, kind: str) -> int:
        """Number of live facts of a given `kind` (verbatim/atomic)."""
        ...

    def candidates(
        self,
        query: str,
        *,
        top_k: int,
        is_reason: bool = False,
        is_temporal: bool = False,
        kind: Optional[str] = None,
        exclude_buckets: Optional[Sequence[int]] = None,
    ) -> CandidatePool:
        """Generate the candidate pool for a query in ONE backend round trip.

        When `kind` is None the whole bank is searched (single-index behaviour).
        When set (``"verbatim"`` / ``"atomic"``) retrieval is scoped to that
        lane — routing, cosines and the topic partition are all kind-local — so
        the union in `MemorySystem` retrieves each lane independently.
        ``exclude_buckets`` closes topic buckets for this call (lane-local ids);
        excluded buckets must never contribute candidates.
        """
        ...

    def partition(self) -> Dict:
        """Topic partition (centroids/buckets) for inspection + labelling."""
        ...

    def topic_exemplars(self, bucket: int, n: int = 5) -> List[str]:
        """Up to `n` representative fact texts for a topic bucket."""
        ...

    def delete_facts(self, fact_ids: Sequence[str]) -> int:
        """Erase specific facts. Returns how many rows were removed.

        Part of the contract because users have to be able to remove a fact
        that is wrong or that they do not want retained; a backend that can
        only append is not a complete memory store. Implementations must also
        clear `superseded_by` on any surviving fact that pointed at a deleted
        one, so removing a bad correction restores the fact it had replaced
        rather than leaving the bank with no current value.
        """
        ...

    def delete(self) -> None:
        """Delete the whole bank (namespace) — GDPR / teardown."""
        ...
