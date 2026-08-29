"""In-memory backend — the original numpy cosine + KMeans behaviour.

This preserves byte-for-byte what `MemorySystem._retrieve` did before the
storage seam was introduced, so the eval harness and benchmark numbers do not
move. It is the default backend and the one tests/eval run on.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from membukkit import telemetry
from membukkit.config import RetrievalConfig
from membukkit.storage.base import Candidate, CandidatePool, FactRecord


class InMemoryBackend:
    """Holds facts/embeddings/partition in RAM (numpy)."""

    def __init__(self, cfg: RetrievalConfig, encoder):
        self._cfg = cfg
        self._encoder = encoder
        self._texts: List[str] = []
        self._times: List[Optional[datetime]] = []
        self._entities: List[List[str]] = []
        self._time_buckets: List[str] = []
        self._kinds: List[str] = []
        self._sources: List[str] = []
        self._speakers: List[str] = []
        self._doc_ids: List[str] = []
        self._doc_names: List[str] = []
        self._source_refs: List[str] = []
        self._superseded_by: List[str] = []
        self._valid_to: List[Optional[datetime]] = []
        self._ids: List[str] = []
        self._id_set: set = set()
        self._embs: Optional[np.ndarray] = None
        self._partition: Optional[Dict] = None
        # Per-kind views for the union: global row indices + a kind-local topic
        # partition, so each lane routes/scores over only its own facts (exact
        # parity with the eval's separate verbatim/atomic banks).
        self._kind_idx: Dict[str, List[int]] = {}
        self._kind_partitions: Dict[str, Dict] = {}
        # Optional BM25 lane, keyed by lane ("" = whole bank). Built lazily on
        # first use and only when cfg.lexical_lane is on; each entry is
        # (n_rows_at_build, index) so appends invalidate it.
        self._bm25: Dict[str, Tuple[int, object]] = {}

    # --------------------------------------------------------- lexical lane
    def _lexical_hits(
        self, query: str, lane: str, rows: Sequence[int], top_k: int
    ) -> List[Tuple[int, float]]:
        """BM25 top-k over `rows`, returned as (position-in-rows, score).

        Only reached when `cfg.lexical_lane` is on. The index is cached per
        lane and rebuilt when the lane's row count changes.
        """
        from membukkit.retrieval.lexical import build_index

        cached = self._bm25.get(lane)
        if cached is None or cached[0] != len(rows):
            index = build_index([self._texts[i] for i in rows])
            if index is None:
                return []
            self._bm25[lane] = (len(rows), index)
        return self._bm25[lane][1].top_k(query, top_k)

    @staticmethod
    def _union_lexical(
        local: List[int], hits: List[Tuple[int, float]]
    ) -> Tuple[List[int], "np.ndarray", int]:
        """Append lexical-only hits to the routed pool, preserving routed order.

        Returns the extended pool, per-candidate BM25 scores aligned to it, and
        how many candidates the lane contributed that routing had missed.
        """
        seen = {int(i) for i in local}
        added = [i for i, _ in hits if int(i) not in seen]
        pool = list(local) + added
        by_pos = {int(i): float(s) for i, s in hits}
        scores = np.asarray([by_pos.get(int(i), 0.0) for i in pool], dtype=np.float64)
        return pool, scores, len(added)

    # ------------------------------------------------------------------ writes
    def clear(self) -> None:
        self._bm25 = {}
        self._texts, self._times = [], []
        self._entities, self._time_buckets = [], []
        self._kinds, self._sources = [], []
        self._speakers = []
        self._doc_ids, self._doc_names, self._source_refs = [], [], []
        self._superseded_by, self._valid_to = [], []
        self._ids, self._id_set = [], set()
        self._embs = None
        self._partition = None
        self._kind_idx = {}
        self._kind_partitions = {}

    def upsert_facts(self, facts: Sequence[FactRecord], on_progress=None) -> int:
        """Append new facts (dedup on id). Embeds ONLY the new ones."""
        from membukkit.progress import emit

        new: List[FactRecord] = []
        n_in = 0
        for f in facts:
            n_in += 1
            fid = f.ensure_id()
            if fid in self._id_set:
                continue
            self._id_set.add(fid)
            new.append(f)
        with telemetry.span(
            "memory.upsert", n_in=n_in, n_new=len(new), n_dedup=n_in - len(new)
        ) as sp:
            telemetry.counter("membukkit.facts.dedup_skipped").add(n_in - len(new))
            if not new:
                return 0

            new_texts = [f.text for f in new]
            with telemetry.timed(
                "memory.embed",
                telemetry.histogram("membukkit.embed.duration"),
                kind="facts",
                count=len(new_texts),
            ):
                from membukkit.progress import encode_with_progress

                new_vecs = encode_with_progress(
                    self._encoder, new_texts, on_progress=on_progress
                )
            new_vecs = np.asarray(new_vecs, dtype=np.float32)
            self._append(new, new_vecs)
            telemetry.set_attributes(sp, total=len(self._texts))
            emit(on_progress, "embed", len(new_texts), len(new_texts), detail="embedded")
            return len(new)

    def _append(self, new: List[FactRecord], new_vecs: np.ndarray) -> None:
        base = len(self._texts)
        self._texts.extend(f.text for f in new)
        self._times.extend(f.timestamp for f in new)
        self._entities.extend(list(f.entities) for f in new)
        self._time_buckets.extend(f.time_bucket for f in new)
        self._kinds.extend(f.kind for f in new)
        self._sources.extend(f.source_session or "" for f in new)
        self._speakers.extend(f.source_speaker or "" for f in new)
        self._doc_ids.extend(f.doc_id for f in new)
        self._doc_names.extend(f.doc_name for f in new)
        self._source_refs.extend(f.source_ref for f in new)
        self._superseded_by.extend(f.superseded_by or "" for f in new)
        self._valid_to.extend(f.valid_to for f in new)
        self._ids.extend(f.id for f in new)
        for offset, f in enumerate(new):
            self._kind_idx.setdefault(f.kind, []).append(base + offset)

        self._embs = new_vecs if self._embs is None else np.vstack([self._embs, new_vecs])
        # Topic partitions (global + per-kind) depend on the embedding set; rebuild lazily.
        self._partition = None
        self._kind_partitions = {}

    def _source_key(self, i: int) -> Tuple[str, str, str]:
        """Identity of the ingested turn a row came from.

        An atomic fact and the verbatim turn it was distilled from carry the
        same key, which is what lets deletion find one from the other.
        """
        return (
            self._doc_ids[i] if i < len(self._doc_ids) else "",
            self._sources[i] if i < len(self._sources) else "",
            self._source_refs[i] if i < len(self._source_refs) else "",
        )

    def orphaned_source_ids(self, fact_ids: Sequence[str]) -> List[str]:
        """Verbatim rows whose only atomic referents are in ``fact_ids``.

        Deleting an atomic fact should take its verbatim source with it, or the
        content the user asked to remove is still sitting in the other lane and
        still retrievable. But one turn can distill into several facts, so a
        verbatim row is only removed once nothing else points at it.
        """
        drop = set(fact_ids) & self._id_set
        if not drop:
            return []
        doomed = {self._source_key(i) for i, f in enumerate(self._ids)
                  if f in drop and self._kinds[i] == "atomic"}
        if not doomed:
            return []
        # A source survives if any atomic fact outside the delete set uses it.
        for i, fid in enumerate(self._ids):
            if fid in drop or self._kinds[i] != "atomic":
                continue
            doomed.discard(self._source_key(i))
        return [
            fid
            for i, fid in enumerate(self._ids)
            if self._kinds[i] == "verbatim" and self._source_key(i) in doomed
        ]

    def source_group_ids(self, fact_ids: Sequence[str]) -> List[str]:
        """Every row, both lanes, sharing an ingested turn with ``fact_ids``.

        The nuclear option behind ``purge_source``: remove the turn and
        everything derived from it, not just the fact that was pointed at.
        """
        drop = set(fact_ids) & self._id_set
        if not drop:
            return []
        keys = {self._source_key(i) for i, f in enumerate(self._ids) if f in drop}
        # A session-level ref ("session:3") owns its turn-level refs too.
        loose = {(d, s, r) for (d, s, r) in keys if "/turn:" not in r}
        return [
            fid
            for i, fid in enumerate(self._ids)
            if self._source_key(i) in keys
            or any(
                self._source_key(i)[:2] == (d, s)
                and self._source_refs[i].startswith(f"{r}/turn:")
                for (d, s, r) in loose
            )
        ]

    def delete_facts(self, fact_ids: Sequence[str]) -> int:
        """Remove facts by id. Returns how many were actually deleted.

        All parallel row structures (texts/times/entities/kinds/provenance/
        vectors) are filtered together so indices stay aligned, the per-kind
        index is rebuilt, and cached topic partitions are invalidated (the
        embedding set changed, so clusters must be recomputed lazily).

        Supersession pointers into the deleted set are repaired: a fact that
        was superseded by a now-deleted fact becomes current again. Without
        that, deleting a wrong correction would leave the store with no
        current value at all, because `is_active_as_of` treats any non-empty
        `superseded_by` as "excluded" without checking the target still exists.
        """
        drop = set(fact_ids) & self._id_set
        if not drop:
            return 0
        keep = [i for i, fid in enumerate(self._ids) if fid not in drop]
        n = len(self._texts)

        def _take(lst, default):
            # Legacy states may have short lists (readers guard with i<len);
            # pad with the same defaults those readers use.
            return [lst[i] if i < len(lst) else default for i in keep]

        self._texts = _take(self._texts, "")
        self._times = _take(self._times, None)
        self._entities = _take(self._entities, [])
        self._time_buckets = _take(self._time_buckets, "unknown")
        self._kinds = _take(self._kinds, "")
        self._sources = _take(self._sources, "")
        self._speakers = _take(self._speakers, "")
        self._doc_ids = _take(self._doc_ids, "")
        self._doc_names = _take(self._doc_names, "")
        self._source_refs = _take(self._source_refs, "")
        self._superseded_by = _take(self._superseded_by, "")
        self._valid_to = _take(self._valid_to, None)
        self._ids = _take(self._ids, "")
        self._id_set = set(self._ids)
        if self._embs is not None:
            self._embs = self._embs[keep] if keep else None
        # Revive facts whose superseding fact just went away.
        for i, sb in enumerate(self._superseded_by):
            if sb and sb in drop:
                self._superseded_by[i] = ""
                self._valid_to[i] = None
        self._kind_idx = {}
        for i, k in enumerate(self._kinds):
            self._kind_idx.setdefault(k, []).append(i)
        self._partition = None
        self._kind_partitions = {}
        self._bm25 = {}  # row positions moved; a cached lexical index is stale
        return n - len(keep)

    def delete_doc_facts(self, doc_id: str) -> int:
        """Remove every fact ingested from the given source document."""
        ids = [fid for fid, d in zip(self._ids, self._doc_ids) if d == doc_id]
        return self.delete_facts(ids)

    def resolve_ids(self, refs: Sequence[str]) -> Tuple[List[str], List[str]]:
        """Map user-facing refs onto stored fact ids.

        Search and receipts show ``mem:<first-12-chars>``, not the full id, so
        anything that erases by id has to accept the form people actually see.
        Returns ``(resolved, unknown)``; an ambiguous prefix raises.
        """
        resolved: List[str] = []
        unknown: List[str] = []
        for raw in refs:
            ref = (raw or "").strip()
            if not ref:
                continue
            if ref in self._id_set:
                resolved.append(ref)
                continue
            probe = ref[4:] if ref.startswith("mem:") else ref
            hits = [f for f in self._ids if f.startswith(probe)] if probe else []
            if len(hits) == 1:
                resolved.append(hits[0])
            elif len(hits) > 1:
                raise ValueError(
                    f"ref {raw!r} is ambiguous ({len(hits)} matches); use the full fact id"
                )
            else:
                unknown.append(raw)
        return list(dict.fromkeys(resolved)), unknown

    def ids_for_source(self, *, doc_id: str = "", source_session: str = "") -> List[str]:
        """Ids of every row, both lanes, from a document or conversation."""
        out: List[str] = []
        for i, fid in enumerate(self._ids):
            if doc_id and (i >= len(self._doc_ids) or self._doc_ids[i] != doc_id):
                continue
            if source_session and (
                i >= len(self._sources) or self._sources[i] != source_session
            ):
                continue
            out.append(fid)
        return out

    def list_rows_for_ids(self, fact_ids: Sequence[str]) -> List[Dict]:
        """Minimal row views for the given ids (used to report what was erased)."""
        want = set(fact_ids)
        return [
            {
                "id": fid,
                "kind": self._kinds[i] if i < len(self._kinds) else "",
                "text": self._texts[i],
            }
            for i, fid in enumerate(self._ids)
            if fid in want
        ]

    def count(self) -> int:
        return len(self._texts)

    def count_kind(self, kind: str) -> int:
        return len(self._kind_idx.get(kind, []))

    def latest_fact_date(self) -> Optional[str]:
        """ISO date (YYYY-MM-DD) of the newest fact timestamp, or None."""
        latest: Optional[datetime] = None
        for ts in self._times:
            if ts is None:
                continue
            naive = ts.replace(tzinfo=None) if getattr(ts, "tzinfo", None) else ts
            if latest is None or naive > latest:
                latest = naive
        return latest.date().isoformat() if latest else None

    # ----------------------------------------------------------------- reads
    def partition(self) -> Dict:
        if self._partition is not None:
            return self._partition
        if self._embs is None or len(self._texts) == 0:
            return {}
        from membukkit.retrieval.buckets import build_topic_partition

        self._partition = build_topic_partition(
            self._embs,
            k=self._cfg.num_buckets,
            k_proto=self._cfg.k_proto,
        )
        return self._partition

    def topic_exemplars(self, bucket: int, n: int = 5, kind: Optional[str] = None) -> List[str]:
        """Up to `n` representative fact texts for a topic bucket.

        With `kind`, the bucket id is lane-local: exemplars come from that
        lane's own partition (so an atomic-lane map shows atomic facts).
        """
        if kind is None:
            part = self.partition()
            ids = part.get("by_bucket", {}).get(bucket, [])
            return [self._texts[i][:150] for i in ids[:n]]
        part = self._kind_partition(kind)
        idxs = self._kind_idx.get(kind, [])
        local = part.get("by_bucket", {}).get(bucket, []) if part else []
        return [self._texts[idxs[j]][:150] for j in local[:n]]

    def _kind_partition(self, kind: str) -> Dict:
        """Topic partition built over ONLY this kind's embeddings (cached)."""
        cached = self._kind_partitions.get(kind)
        if cached is not None:
            return cached
        idxs = self._kind_idx.get(kind, [])
        if self._embs is None or not idxs:
            return {}
        from membukkit.retrieval.buckets import build_topic_partition

        part = build_topic_partition(
            self._embs[idxs],
            k=self._cfg.num_buckets,
            k_proto=self._cfg.k_proto,
        )
        self._kind_partitions[kind] = part
        return part

    def lane_view(self, kind: Optional[str] = None) -> Dict:
        """Inspection view of one lane (or the whole bank when kind is None).

        Returns kind-local, position-aligned lists — {"ids", "texts", "sources",
        "labels", "k_eff"} — so callers (eval interventions, audits) can map a
        fact id or ingest session to the topic bucket it landed in. Labels come
        from the same cached partition routing uses, so the view is faithful to
        what retrieval actually does.
        """
        if kind is None:
            part = self.partition()
            idxs = list(range(len(self._texts)))
        else:
            part = self._kind_partition(kind)
            idxs = self._kind_idx.get(kind, [])
        labels = part.get("labels") if part else None
        if labels is None or not idxs:
            return {}
        return {
            "ids": [self._ids[g] for g in idxs],
            "texts": [self._texts[g] for g in idxs],
            "sources": [self._sources[g] if g < len(self._sources) else "" for g in idxs],
            "labels": [int(labels[j]) for j in range(len(idxs))],
            "k_eff": int(part.get("k_eff", 0)),
        }

    def _candidate(self, idx: int, cosine: float = 0.0, lexical: float = 0.0) -> Candidate:
        return Candidate(
            text=self._texts[idx],
            timestamp=self._times[idx] if idx < len(self._times) else None,
            cosine=cosine,
            lexical=lexical,
            entities=self._entities[idx] if idx < len(self._entities) else [],
            time_bucket=self._time_buckets[idx] if idx < len(self._time_buckets) else "unknown",
            kind=self._kinds[idx] if idx < len(self._kinds) else "",
            id=self._ids[idx] if idx < len(self._ids) else "",
            doc_id=self._doc_ids[idx] if idx < len(self._doc_ids) else "",
            doc_name=self._doc_names[idx] if idx < len(self._doc_names) else "",
            source_ref=self._source_refs[idx] if idx < len(self._source_refs) else "",
            superseded_by=self._superseded_by[idx] if idx < len(self._superseded_by) else "",
            valid_to=self._valid_to[idx] if idx < len(self._valid_to) else None,
        )

    def list_atomic_rows(self) -> List[Dict]:
        """Atomic facts for supersession linking."""
        out = []
        for i, kind in enumerate(self._kinds):
            if kind != "atomic":
                continue
            out.append(
                {
                    "id": self._ids[i],
                    "text": self._texts[i],
                    "timestamp": self._times[i] if i < len(self._times) else None,
                    "entities": self._entities[i] if i < len(self._entities) else [],
                    "superseded_by": self._superseded_by[i]
                    if i < len(self._superseded_by)
                    else "",
                }
            )
        return out

    def supersede(
        self, pairs: Sequence[Tuple[str, str]], when: Optional[datetime] = None
    ) -> None:
        """Mark old facts superseded by newer ones (in-memory)."""
        when = when or datetime.now(timezone.utc)
        for old_id, new_id in pairs:
            try:
                g = self._ids.index(old_id)
            except ValueError:
                continue
            while len(self._superseded_by) < len(self._ids):
                self._superseded_by.append("")
            while len(self._valid_to) < len(self._ids):
                self._valid_to.append(None)
            self._superseded_by[g] = new_id or "updated"
            self._valid_to[g] = when

    def set_valid_to(self, fact_id: str, when: datetime) -> None:
        try:
            g = self._ids.index(fact_id)
        except ValueError:
            return
        while len(self._valid_to) < len(self._ids):
            self._valid_to.append(None)
        self._valid_to[g] = when

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
        """Replicates the original `_retrieve` candidate-generation path.

        With `kind` set, retrieval is scoped to that lane: routing and cosines
        run over only the kind's rows against a kind-local topic partition, then
        local indices are mapped back to global rows for the returned candidates.
        `exclude_buckets` closes topic buckets for this call (topic mode only).
        """
        if kind is not None:
            return self._candidates_kind(
                query, top_k, is_reason, is_temporal, kind, exclude_buckets
            )

        cfg = self._cfg
        fe = self._embs
        if fe is None or len(self._texts) == 0:
            return CandidatePool([], {}, False)
        with telemetry.timed(
            "memory.embed", telemetry.histogram("membukkit.embed.duration"), kind="query", count=1
        ):
            qe = np.asarray(self._encoder.encode(query, normalize=True), dtype=np.float32)

        cand_idx: List[int]
        cosines: Optional[np.ndarray] = None
        has_cosine = False

        if cfg.bucket_mode == "multiaxis":
            from membukkit.retrieval.buckets import build_multiaxis_partition, route_multiaxis

            budget = cfg.scan_budget_reason if is_reason else cfg.scan_budget
            part = build_multiaxis_partition(self._texts, self._times, fe, k=cfg.num_buckets)
            cand_idx, trace = route_multiaxis(
                part,
                query,
                qe,
                fe,
                budget=budget,
                temporal=is_temporal,
                rerank_cap=cfg.rerank_cap,
            )
        elif cfg.bucket_mode == "topic":
            from membukkit.retrieval.buckets import route_topic

            partition = self.partition()
            budget_temporal = cfg.scan_budget_temporal or cfg.scan_budget
            budget = budget_temporal if is_temporal else cfg.scan_budget
            cand_idx, trace = route_topic(partition, qe, budget=budget, exclude=exclude_buckets)

            select = cfg.select
            rcap = cfg.rerank_cap
            if select == "cosine" and rcap and len(cand_idx) > rcap:
                rcos = fe[cand_idx] @ qe
                cand_idx = [cand_idx[j] for j in np.argsort(rcos)[::-1][:rcap]]
            elif select in ("hybrid", "none"):
                # "none" needs cosines attached: the pipeline orders by them
                # directly (plain-cosine arm, cross-encoder skipped entirely).
                cosines = fe[cand_idx] @ qe
                has_cosine = True
        else:
            cos = fe @ qe
            cand_idx = list(np.argsort(cos)[::-1][: min(cfg.candidate_pool, len(self._texts))])
            trace = {"scan_frac": 1.0, "n_facts": len(self._texts), "n_scanned": len(self._texts)}

        lex: Optional[np.ndarray] = None
        if cfg.lexical_lane:
            rows = list(range(len(self._texts)))
            hits = self._lexical_hits(query, "", rows, cfg.lexical_top_k)
            if hits:
                cand_idx, lex, n_added = self._union_lexical(cand_idx, hits)
                cosines = fe[cand_idx] @ qe
                has_cosine = True
                trace = {**trace, "lexical_added": n_added, "lexical_scanned": len(rows)}

        pool = [
            self._candidate(
                idx,
                float(cosines[j]) if cosines is not None else 0.0,
                float(lex[j]) if lex is not None else 0.0,
            )
            for j, idx in enumerate(cand_idx)
        ]
        return CandidatePool(
            candidates=pool, trace=trace, has_cosine=has_cosine, has_lexical=lex is not None
        )

    def _candidates_kind(
        self,
        query: str,
        top_k: int,
        is_reason: bool,
        is_temporal: bool,
        kind: str,
        exclude_buckets: Optional[Sequence[int]] = None,
    ) -> CandidatePool:
        """Kind-scoped candidate generation (verbatim OR atomic lane).

        Mirrors the topic-mode path exactly, but over the kind's own embeddings
        and topic partition — reproducing the eval's per-bank retrieval.
        """
        cfg = self._cfg
        idxs = self._kind_idx.get(kind, [])
        if self._embs is None or not idxs:
            return CandidatePool([], {}, False)
        sub = self._embs[idxs]
        with telemetry.timed(
            "memory.embed", telemetry.histogram("membukkit.embed.duration"), kind="query", count=1
        ):
            qe = np.asarray(self._encoder.encode(query, normalize=True), dtype=np.float32)

        cosines: Optional[np.ndarray] = None
        has_cosine = False

        if cfg.bucket_mode == "topic":
            from membukkit.retrieval.buckets import route_topic

            partition = self._kind_partition(kind)
            budget_temporal = cfg.scan_budget_temporal or cfg.scan_budget
            budget = budget_temporal if is_temporal else cfg.scan_budget
            local, trace = route_topic(partition, qe, budget=budget, exclude=exclude_buckets)

            select = cfg.select
            rcap = cfg.rerank_cap
            if select == "cosine" and rcap and len(local) > rcap:
                rcos = sub[local] @ qe
                local = [local[j] for j in np.argsort(rcos)[::-1][:rcap]]
            elif select in ("hybrid", "none"):
                # "none" needs cosines attached: the pipeline orders by them
                # directly (plain-cosine arm, cross-encoder skipped entirely).
                cosines = sub[local] @ qe
                has_cosine = True
        else:
            cos = sub @ qe
            local = list(np.argsort(cos)[::-1][: min(cfg.candidate_pool, len(idxs))])
            trace = {"scan_frac": 1.0, "n_facts": len(idxs), "n_scanned": len(idxs)}

        lex: Optional[np.ndarray] = None
        if cfg.lexical_lane:
            hits = self._lexical_hits(query, kind, idxs, cfg.lexical_top_k)
            if hits:
                local, lex, n_added = self._union_lexical(local, hits)
                cosines = sub[local] @ qe
                has_cosine = True
                trace = {**trace, "lexical_added": n_added, "lexical_scanned": len(idxs)}

        pool = [
            self._candidate(
                idxs[loc],
                float(cosines[j]) if cosines is not None else 0.0,
                float(lex[j]) if lex is not None else 0.0,
            )
            for j, loc in enumerate(local)
        ]
        return CandidatePool(
            candidates=pool, trace=trace, has_cosine=has_cosine, has_lexical=lex is not None
        )

    # ------------------------------------------------------------ persistence
    def to_state(self) -> Dict:
        """Serializable snapshot (used by LocalStore). Vectors returned separately."""
        rows = []
        for i in range(len(self._texts)):
            rows.append(
                {
                    "id": self._ids[i],
                    "text": self._texts[i],
                    "timestamp": self._times[i].isoformat() if self._times[i] else None,
                    "entities": self._entities[i],
                    "time_bucket": self._time_buckets[i],
                    "kind": self._kinds[i],
                    "source_session": self._sources[i],
                    "source_speaker": self._speakers[i] if i < len(self._speakers) else "",
                    "doc_id": self._doc_ids[i] if i < len(self._doc_ids) else "",
                    "doc_name": self._doc_names[i] if i < len(self._doc_names) else "",
                    "source_ref": self._source_refs[i] if i < len(self._source_refs) else "",
                    "superseded_by": self._superseded_by[i]
                    if i < len(self._superseded_by)
                    else "",
                    "valid_to": self._valid_to[i].isoformat()
                    if i < len(self._valid_to) and self._valid_to[i]
                    else None,
                }
            )
        return {"facts": rows, "vectors": self._embs}

    def from_state(self, rows: List[Dict], vectors: Optional[np.ndarray]) -> None:
        """Restore a snapshot produced by `to_state` (replaces current contents)."""
        self.clear()
        if not rows:
            return
        self._texts = [r["text"] for r in rows]
        self._times = [
            datetime.fromisoformat(r["timestamp"]) if r.get("timestamp") else None for r in rows
        ]
        self._entities = [list(r.get("entities") or []) for r in rows]
        self._time_buckets = [r.get("time_bucket", "unknown") for r in rows]
        self._kinds = [r.get("kind", "atomic") for r in rows]
        self._sources = [r.get("source_session", "") or "" for r in rows]
        self._speakers = [r.get("source_speaker", "") or "" for r in rows]
        self._doc_ids = [r.get("doc_id", "") for r in rows]
        self._doc_names = [r.get("doc_name", "") for r in rows]
        self._source_refs = [r.get("source_ref", "") for r in rows]
        self._superseded_by = [r.get("superseded_by", "") or "" for r in rows]
        self._valid_to = [
            datetime.fromisoformat(r["valid_to"]) if r.get("valid_to") else None for r in rows
        ]
        self._ids = [r["id"] for r in rows]
        self._id_set = set(self._ids)
        self._embs = np.asarray(vectors, dtype=np.float32) if vectors is not None else None
        for i, k in enumerate(self._kinds):
            self._kind_idx.setdefault(k, []).append(i)

    # -------------------------------------------------------------- browsing
    def facts_page(
        self,
        offset: int = 0,
        limit: int = 50,
        kind: Optional[str] = None,
        bucket: Optional[int] = None,
    ) -> Dict:
        """Paginated fact browser (service/GUI surface).

        When `bucket` is given, rows are filtered to that topic bucket of the
        requested lane (kind-local partition; global partition if kind is None).
        """
        idxs = self._kind_idx.get(kind, []) if kind else list(range(len(self._texts)))
        if bucket is not None:
            part = self._kind_partition(kind) if kind else self.partition()
            labels = part.get("labels") if part else None
            if labels is not None:
                idxs = [g for j, g in enumerate(idxs) if int(labels[j]) == bucket]
        total = len(idxs)
        window = idxs[offset : offset + limit]
        facts = []
        for g in window:
            facts.append(
                {
                    "id": self._ids[g],
                    "text": self._texts[g],
                    "timestamp": self._times[g].isoformat() if self._times[g] else None,
                    "kind": self._kinds[g],
                    "entities": self._entities[g],
                    "source_session": self._sources[g],
                    "doc_id": self._doc_ids[g] if g < len(self._doc_ids) else "",
                    "doc_name": self._doc_names[g] if g < len(self._doc_names) else "",
                    "source_ref": self._source_refs[g] if g < len(self._source_refs) else "",
                    "superseded_by": self._superseded_by[g]
                    if g < len(self._superseded_by)
                    else "",
                    "valid_to": self._valid_to[g].isoformat()
                    if g < len(self._valid_to) and self._valid_to[g]
                    else None,
                    "status": "superseded"
                    if (g < len(self._superseded_by) and self._superseded_by[g])
                    else "current",
                }
            )
        return {"total": total, "offset": offset, "facts": facts}

    def get_fact(self, fact_id: str) -> Optional[Dict]:
        """Look up one fact row by id (provenance drill-down)."""
        try:
            g = self._ids.index(fact_id)
        except ValueError:
            return None
        superseded = self._superseded_by[g] if g < len(self._superseded_by) else ""
        return {
            "id": self._ids[g],
            "text": self._texts[g],
            "timestamp": self._times[g].isoformat() if self._times[g] else None,
            "kind": self._kinds[g],
            "entities": self._entities[g],
            "source_session": self._sources[g],
            "source_speaker": self._speakers[g] if g < len(self._speakers) else "",
            "doc_id": self._doc_ids[g] if g < len(self._doc_ids) else "",
            "doc_name": self._doc_names[g] if g < len(self._doc_names) else "",
            "source_ref": self._source_refs[g] if g < len(self._source_refs) else "",
            "superseded_by": superseded,
            "valid_to": self._valid_to[g].isoformat()
            if g < len(self._valid_to) and self._valid_to[g]
            else None,
            "status": "superseded" if superseded else "current",
        }

    def delete(self) -> None:
        self.clear()
