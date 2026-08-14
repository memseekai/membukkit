"""Turbopuffer storage backend — persistent, namespace-per-owner memory.

One namespace per memory owner (``mem_<owner>``). Facts are stored with their
embedding (f16) plus filterable metadata; retrieval is a single ``multi_query``
(dense ANN lane + optional BM25 lexical lane) fused server-side via RRF, then
the local cross-encoder reranks the pool. Each fact ``kind`` (verbatim/atomic)
gets its OWN topic partition (centroids), stored as a sentinel
``__partition__:<kind>`` document in the same namespace and cached per worker,
so each union lane routes over only its own facts.

The SDK is imported lazily so the package imports without ``turbopuffer``
installed. All SDK calls funnel through tiny ``_q``/``_write`` seams so they can
be mocked in tests.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from membukkit import telemetry
from membukkit.config import RetrievalConfig, StorageConfig
from membukkit.storage.base import Candidate, CandidatePool, FactRecord
from membukkit.time_utils import TS_UNKNOWN, is_unknown_ts, parse_datetime

logger = logging.getLogger(__name__)

_META_ID = "__partition__"  # legacy (pre per-kind) sentinel; still read for back-compat
_META_SENTINEL = "__meta__"  # superseded_by value that hides meta docs from fact queries
_UPSERT_BATCH = 10_000
_MAX_TOP_K = 1_200  # Turbopuffer's per-query top_k ceiling; scans page under this.
_DEFAULT_KIND = "atomic"  # kind used for non-union / Protocol-level partition()


def _meta_id(kind: str) -> str:
    """Sentinel doc id for a kind's topic partition (one partition per kind).

    Verbatim (raw turns) and atomic (distilled facts) occupy differently shaped
    semantic spaces, so each gets its own centroid set — mirroring the in-memory
    backend's per-kind partitions — for optimal routing on each union lane.
    """
    return f"{_META_ID}:{kind}"


class TurbopufferBackend:
    """Persistent memory bank backed by a single Turbopuffer namespace."""

    def __init__(self, cfg: RetrievalConfig, encoder, storage: StorageConfig):
        if not storage.namespace:
            raise ValueError("StorageConfig.namespace is required for the Turbopuffer backend")
        self._cfg = cfg
        self._encoder = encoder
        self._storage = storage
        self._dim: Optional[int] = None
        self._client = None
        self._namespace = None
        # One cached partition per kind (centroids/sizes/version). Absence of a
        # key means "not loaded yet"; a stored None means "loaded, none exists".
        self._partitions: Dict[str, Optional[Dict]] = {}
        self._count_cache: Optional[int] = None

    # ------------------------------------------------------------- SDK seams
    @property
    def _ns(self):
        if self._namespace is None:
            import turbopuffer

            self._client = turbopuffer.Turbopuffer(
                region=self._storage.region,
                api_key=self._storage.api_key,
            )
            assert self._storage.namespace is not None
            self._namespace = self._client.namespace(self._storage.namespace)
        return self._namespace

    def _write(self, **kwargs):
        return self._ns.write(**kwargs)

    def _q(self, **kwargs):
        return self._ns.query(**kwargs)

    def _multi_q(self, queries, rerank_by):
        return self._ns.multi_query(queries=queries, rerank_by=rerank_by)

    # --------------------------------------------------------------- schema
    def _vector_type(self) -> str:
        dt = (self._storage.vector_dtype or "f16").lower()
        dim = self._dim or 768
        return f"[{dim}]{'f16' if dt == 'f16' else 'f32'}"

    def _schema(self) -> Dict:
        return {
            "vector": {"type": self._vector_type(), "ann": True},
            "text": {"type": "string", "full_text_search": True},
            "ts": {"type": "datetime"},
            "kind": {"type": "string"},
            "topic_bucket": {"type": "int"},
            "entities": {"type": "[]string"},
            "time_bucket": {"type": "string"},
            "source_session": {"type": "string"},
            "source_speaker": {"type": "string"},
            "tag": {"type": "string"},
            "subject": {"type": "string"},
            "superseded_by": {"type": "string"},
            "valid_to": {"type": "datetime"},
            "partition_version": {"type": "int"},
        }

    # ---------------------------------------------------------------- writes
    def clear(self) -> None:
        """Forget cached state. (Does NOT delete the namespace — use delete().)"""
        self._partitions = {}
        self._count_cache = None

    def _partition_kinds(self) -> Tuple[str, ...]:
        """Kinds that get their own partition. Under union both lanes are stored
        and each is partitioned separately; otherwise a single atomic partition."""
        if self._cfg.union:
            return tuple(self._cfg.union_lanes or ("verbatim", "atomic"))
        return (_DEFAULT_KIND,)

    def _existing_ids(self, ids: Sequence[str]) -> set:
        """Which of `ids` already exist (so we embed only genuinely new facts)."""
        if not ids:
            return set()
        try:
            res = self._q(
                rank_by=("id", "asc"),
                filters=["id", "In", list(ids)],
                top_k=len(ids),
                include_attributes=[],
            )
            return {_row_get(r, "id") for r in _rows(res)}
        except Exception as e:  # cold namespace / first write
            logger.debug("existing_ids lookup failed (treating as empty): %s", e)
            return set()

    def upsert_facts(self, facts: Sequence[FactRecord], on_progress=None) -> int:
        from membukkit.progress import emit, encode_with_progress

        # De-dup within the batch and against the namespace.
        seen, deduped = set(), []
        for f in facts:
            fid = f.ensure_id()
            if fid in seen:
                continue
            seen.add(fid)
            deduped.append(f)
        existing = self._existing_ids([f.id for f in deduped])
        new = [f for f in deduped if f.id not in existing]
        upsert_span = telemetry.span(
            "memory.upsert", n_in=len(facts), n_new=len(new), n_dedup=len(facts) - len(new)
        )
        telemetry.counter("membukkit.facts.dedup_skipped").add(len(facts) - len(new))
        if not new:
            return 0

        with upsert_span:
            with telemetry.timed(
                "memory.embed",
                telemetry.histogram("membukkit.embed.duration"),
                kind="facts",
                count=len(new),
            ):
                vecs = np.asarray(
                    encode_with_progress(
                        self._encoder,
                        [f.text for f in new],
                        on_progress=on_progress,
                    ),
                    dtype=np.float32,
                )
            if vecs.ndim == 1:
                vecs = vecs[None, :]
            self._dim = vecs.shape[1]
            n = self._write_new(new, vecs)
            emit(on_progress, "embed", len(new), len(new), detail="embedded")
            return n

    def _write_new(self, new, vecs) -> int:
        # Assign topic buckets against each fact's OWN kind partition (nearest
        # centroid, no refit). The batch mixes verbatim + atomic (one ingest
        # writes both lanes), so route each kind through its own centroids.
        from membukkit.retrieval.buckets import assign_nearest

        buckets = np.full(len(new), -1, dtype=np.int64)
        pversions = np.zeros(len(new), dtype=np.int64)
        by_kind: Dict[str, List[int]] = {}
        for i, f in enumerate(new):
            by_kind.setdefault(f.kind, []).append(i)
        for kind, idxs in by_kind.items():
            part = self._load_partition(kind)
            if part is None or part.get("centroids") is None:
                continue
            sub = vecs[idxs]
            labels = assign_nearest(part["centroids"], sub)
            ver = int(part.get("version", 0))
            for j, gi in enumerate(idxs):
                buckets[gi] = int(labels[j])
                pversions[gi] = ver

        rows = []
        for f, v, b, pv in zip(new, vecs, buckets, pversions):
            rows.append(
                {
                    "id": f.id,
                    "vector": v.tolist(),
                    "text": f.text,
                    "ts": f.timestamp or TS_UNKNOWN,
                    "kind": f.kind,
                    "topic_bucket": int(b),
                    "entities": list(f.entities),
                    "time_bucket": f.time_bucket,
                    "source_session": f.source_session or "",
                    "source_speaker": f.source_speaker or "",
                    "tag": f.tag,
                    "subject": f.subject or "",
                    "superseded_by": "",
                    "partition_version": int(pv),
                }
            )

        # Seed the count from the server BEFORE writing: afterwards the server
        # total already includes the new rows and incrementing would double-add.
        self.count()
        for i in range(0, len(rows), _UPSERT_BATCH):
            self._write(
                upsert_rows=rows[i : i + _UPSERT_BATCH],
                distance_metric="cosine_distance",
                schema=self._schema(),
            )
        if self._count_cache is not None:
            self._count_cache += len(new)
        return len(new)

    def patch_facts(self, ids: Sequence[str], attrs: Dict) -> None:
        """Uniform scalar/array patch (never vectors). Used for supersede / re-label."""
        if not ids:
            return
        self._patch_rows([{"id": i, **attrs} for i in ids])

    def _patch_rows(self, rows: List[Dict]) -> None:
        for j in range(0, len(rows), _UPSERT_BATCH):
            self._write(patch_rows=rows[j : j + _UPSERT_BATCH])

    def supersede(self, pairs: Sequence[Tuple[str, str]], when: Optional[datetime] = None) -> None:
        """Mark old facts as superseded by newer ones (knowledge-update).

        `pairs` = [(old_id, new_id), ...]. Patches scalars only — the vector is
        untouched — so this is cheap and never re-embeds. Default fact queries
        filter ``superseded_by = ''`` and so stop returning the stale facts.
        """
        when = when or datetime.now(timezone.utc)
        rows = [
            {"id": old, "superseded_by": new, "valid_to": when, "tag": "UPDATED"}
            for old, new in pairs
            if old
        ]
        with telemetry.span("memory.supersede", n=len(rows)):
            self._patch_rows(rows)

    def count(self) -> int:
        if self._count_cache is not None:
            return self._count_cache
        try:
            res = self._q(aggregate_by={"n": ("Count",)}, filters=["superseded_by", "Eq", ""])
            self._count_cache = int(_agg_get(res, "n"))
            return self._count_cache
        except Exception as e:
            # Not cached: a transient failure must not pin the count (and the
            # scan-budget target derived from it) at 0 for the process lifetime.
            logger.debug("count aggregate failed (not cached): %s", e)
            return 0

    def count_kind(self, kind: str) -> int:
        try:
            res = self._q(
                aggregate_by={"n": ("Count",)},
                filters=["And", [["superseded_by", "Eq", ""], ["kind", "Eq", kind]]],
            )
            return int(_agg_get(res, "n"))
        except Exception as e:
            logger.debug("count_kind aggregate failed: %s", e)
            return 0

    # ---------------------------------------------------------------- reads
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
        from membukkit.retrieval.query_filters import build_filter, query_entities, query_time_range

        n = self.count()
        if n == 0:
            return CandidatePool([], {"backend": "turbopuffer"}, False)

        if exclude_buckets and (self._cfg.retrieval_mode or "gated").lower() != "gated":
            raise ValueError(
                "exclude_buckets requires gated retrieval (retrieval_mode='gated'); "
                "the open/ANN-first mode cannot enforce bucket exclusion."
            )

        with telemetry.timed(
            "memory.embed", telemetry.histogram("membukkit.embed.duration"), kind="query", count=1
        ):
            qe = np.asarray(self._encoder.encode(query, normalize=True), dtype=np.float32).ravel()

        entities = query_entities(query)
        time_range = query_time_range(query) if is_temporal else None

        # For a kind-scoped lane, route/budget over ONLY that kind's facts so the
        # scan-fraction target isn't skewed by the other lane's size.
        part = self._load_partition(kind or _DEFAULT_KIND)
        n_lane = self.count_kind(kind) if kind else n

        # Routing / scan-budget trace (computed for BOTH modes from cached centroids).
        opened_ids, opened_trace, scan_frac = self._route(
            qe, is_reason, is_temporal, n_lane, part, exclude=exclude_buckets
        )
        gated = (self._cfg.retrieval_mode or "gated").lower() == "gated"

        # gated: entity/time clauses are OR'd with the bucket clause so they BROADEN
        # beyond the opened buckets (the old multiaxis union). open: ANN already
        # spans the whole bank, so an entity ContainsAny would only RESTRICT it and
        # hurt recall — we keep just the explicit-date scope there. Bucket -1
        # (facts written before a partition existed, not yet relabelled) is always
        # opened — unlabelled facts must never be unreachable.
        if exclude_buckets:
            if not opened_ids:
                # Everything reachable is excluded: return an empty pool rather
                # than falling through to an unfiltered (leaking) search.
                trace = {
                    "buckets": [],
                    "scan_frac": 0.0,
                    "n_facts": n_lane,
                    "n_scanned": 0,
                    "k_total": (part or {}).get("k_eff", 0),
                    "backend": "turbopuffer",
                    "perf": {},
                    "mode": "gated",
                    "excluded_buckets": sorted(int(b) for b in exclude_buckets),
                }
                return CandidatePool([], trace, False)
            # Entity/time broadening could reopen excluded buckets; exclusion is
            # a hard scope, so it wins over the recall-side OR clauses.
            entities = []
        flt = build_filter(
            topic_buckets=[*opened_ids, -1] if (gated and opened_ids) else None,
            entities=(entities or None) if gated else None,
            time_range=time_range,
            live_only=True,
            kind=kind,
        )

        pool_size = max(top_k, self._cfg.pool_size)
        cands, perf, has_cosine = self._search(query, qe, flt, pool_size, kind=kind)

        if not gated:
            # ANN-first: buckets are descriptive only — derive the trace post-hoc.
            opened_trace, scan_frac = self._posthoc_buckets(cands, n_lane, part)

        trace = {
            "buckets": opened_trace,
            "scan_frac": scan_frac if gated else (len(cands) / n_lane if n_lane else 0.0),
            "n_facts": n_lane,
            "n_scanned": len(cands),
            "k_total": (part or {}).get("k_eff", 0),
            "backend": "turbopuffer",
            "perf": perf,
            "mode": "gated" if gated else "open",
        }
        return CandidatePool(candidates=cands, trace=trace, has_cosine=has_cosine)

    def _route(
        self,
        qe,
        is_reason: bool,
        is_temporal: bool,
        n: int,
        part: Optional[Dict],
        exclude: Optional[Sequence[int]] = None,
    ):
        if not part or part.get("centroids") is None:
            return [], [], 0.0
        from membukkit.retrieval.buckets import rank_buckets

        if is_temporal:
            budget = self._cfg.scan_budget_temporal or self._cfg.scan_budget
        elif is_reason:
            budget = self._cfg.scan_budget_reason
        else:
            budget = self._cfg.scan_budget
        return rank_buckets(
            part["centroids"], part.get("sizes", {}), qe, budget, n, exclude=exclude
        )

    def _search(self, query: str, qe, flt, pool_size: int, kind: Optional[str] = None):
        """One round trip: dense ANN (+ optional BM25), server-side RRF. Returns
        (candidates, perf, has_cosine)."""
        inc = ["text", "ts", "kind", "topic_bucket", "entities", "time_bucket"]
        vec_q = {
            "rank_by": ("vector", "ANN", qe.tolist()),
            "top_k": pool_size,
            "include_attributes": inc,
        }
        if flt is not None:
            vec_q["filters"] = flt
        lanes = "vector+bm25" if (self._cfg.bm25_lane and query.strip()) else "vector"

        with telemetry.span(
            "tpuf.search", lanes=lanes, pool_size=pool_size, kind=kind or "all"
        ) as sp:
            if lanes == "vector+bm25":
                bm_q = {
                    "rank_by": ("text", "BM25", query),
                    "top_k": pool_size,
                    "include_attributes": inc,
                }
                if flt is not None:
                    bm_q["filters"] = flt
                res = self._multi_q([vec_q, bm_q], rerank_by=("RRF",))
                rows = _multi_rows(res)
                # Server already fused; use descending rank as the "cosine" lane for the
                # final RRF against the cross-encoder.
                cands = [
                    self._to_candidate(r, cosine=float(len(rows) - i)) for i, r in enumerate(rows)
                ]
            else:
                res = self._q(**vec_q)
                rows = _rows(res)
                cands = [self._to_candidate(r, cosine=_similarity(r)) for r in rows]

            perf = _perf(res)
            self._record_db_perf(sp, lanes, len(cands), perf, kind=kind)
            return cands, perf, True

    @staticmethod
    def _record_db_perf(
        sp, lanes: str, n_returned: int, perf: Dict, kind: Optional[str] = None
    ) -> None:
        """Surface Turbopuffer's perf block onto the span + db.query.duration metric."""
        cache_temp = perf.get("cache_temperature") or perf.get("cache_hit_ratio")
        server_ms = perf.get("server_total_ms") or perf.get("query_execution_ms")
        bytes_q = perf.get("billable_logical_bytes_queried") or perf.get("bytes_queried")
        telemetry.set_attributes(
            sp,
            n_returned=n_returned,
            cache_temperature=str(cache_temp) if cache_temp else "",
            server_total_ms=server_ms or 0,
            bytes_queried=bytes_q or 0,
        )
        if server_ms is not None:
            labels = {"lanes": lanes, "kind": kind or "all"}
            if cache_temp:
                labels["cache_temperature"] = str(cache_temp)
            telemetry.histogram("membukkit.db.query.duration").record(float(server_ms), labels)

    def _to_candidate(self, row, cosine: float) -> Candidate:
        ts = parse_datetime(_row_get(row, "ts"))
        if is_unknown_ts(ts):
            ts = None  # storage sentinel, not a real 1970 date
        return Candidate(
            text=_row_get(row, "text") or "",
            timestamp=ts,
            cosine=cosine,
            topic_bucket=int(_row_get(row, "topic_bucket") or -1),
            entities=list(_row_get(row, "entities") or []),
            time_bucket=_row_get(row, "time_bucket") or "unknown",
            kind=_row_get(row, "kind") or "",
            id=_row_get(row, "id") or "",
        )

    def _posthoc_buckets(self, cands: List[Candidate], n: int, part: Optional[Dict]):
        sizes = (part or {}).get("sizes", {})
        counts: Dict[int, int] = {}
        for c in cands:
            counts[c.topic_bucket] = counts.get(c.topic_bucket, 0) + 1
        opened = [
            {"bucket": b, "hits": h, "size": int(sizes.get(b, 0))}
            for b, h in sorted(counts.items(), key=lambda x: -x[1])
            if b >= 0
        ]
        return opened, (len(cands) / n if n else 0.0)

    # ------------------------------------------------------------ partition
    def partition(self) -> Dict:
        """Protocol-level (kind-agnostic) partition — used for bucket labelling.

        Returns the default-kind partition (atomic), building it on demand. The
        per-kind partitions used by retrieval live behind ``_load_partition``.
        """
        kind = _DEFAULT_KIND
        part = self._load_partition(kind)
        if part is not None:
            return part
        return self._build_partition(kind)

    def _load_partition(self, kind: str = _DEFAULT_KIND) -> Optional[Dict]:
        if kind in self._partitions:
            return self._partitions[kind]
        part = self._load_partition_doc(_meta_id(kind))
        if part is None:
            # Back-compat: a namespace built before per-kind partitions has one
            # legacy ``__partition__`` doc (atomic-only). Fall back to it so
            # existing banks keep routing until the next per-kind (re)build.
            part = self._load_partition_doc(_META_ID)
        self._partitions[kind] = part
        return part

    def _load_partition_doc(self, doc_id: str) -> Optional[Dict]:
        try:
            res = self._q(
                rank_by=("id", "asc"),
                filters=["id", "Eq", doc_id],
                top_k=1,
                include_attributes=True,  # "payload" may not be in the schema yet
            )
            rows = _rows(res)
            if not rows:
                return None
            return _decode_partition(_row_get(rows[0], "payload"))
        except Exception as e:
            logger.debug("partition load failed (%s): %s", doc_id, e)
            return None

    def _save_partition(self, part: Dict, kind: str = _DEFAULT_KIND) -> None:
        payload = _encode_partition(part)
        dim = part["centroids"].shape[1]
        self._write(
            upsert_rows=[
                {
                    "id": _meta_id(kind),
                    "vector": [0.0] * dim,
                    "text": "",
                    "kind": kind,
                    "superseded_by": _META_SENTINEL,  # hidden from fact queries
                    "payload": payload,
                    "partition_version": int(part.get("version", 0)),
                }
            ],
            distance_metric="cosine_distance",
            # payload is the (large) encoded centroids blob — stored, never
            # filtered; filterable attrs are capped at 4 KiB, this is ~96 KiB.
            schema={**self._schema(), "payload": {"type": "string", "filterable": False}},
        )
        self._partitions[kind] = part

    def _build_partition(
        self, kind: str = _DEFAULT_KIND, sample: int = 50_000, relabel_sample: bool = True
    ) -> Dict:
        """Sample this kind's vectors, KMeans, persist centroids, relabel samples.

        `relabel_sample` patches the sampled rows' `topic_bucket` so facts written
        before the first partition existed (bucket -1) become routable immediately;
        `recluster()` passes False because it relabels every row right after.
        """
        ids, vecs = self._sample_vectors(sample, kind=kind)
        if len(vecs) == 0:
            return {}
        from membukkit.retrieval.buckets import build_topic_partition

        raw = build_topic_partition(vecs, k=self._cfg.num_buckets, k_proto=self._cfg.k_proto)
        cents = raw["centroids_norm"]
        prev = self._partitions.get(kind) or {}  # cache-only: version bump, no round trip
        part = {
            "centroids": np.asarray(cents, dtype=np.float32),
            "k_eff": int(raw["k_eff"]),
            "version": int(prev.get("version", 0)) + 1,
            "sizes": {},
            "n_at_build": self.count_kind(kind),
        }
        # Assign the sampled vectors to get bucket sizes; a full re-label of
        # rows outside the sample is recluster()'s job.
        from membukkit.retrieval.buckets import assign_nearest

        labels = assign_nearest(cents, vecs)
        sizes: Dict[int, int] = {}
        for b in labels:
            sizes[int(b)] = sizes.get(int(b), 0) + 1
        part["sizes"] = sizes
        self._save_partition(part, kind=kind)
        if relabel_sample:
            version = int(part["version"])
            self._patch_rows(
                [
                    {"id": i, "topic_bucket": int(b), "partition_version": version}
                    for i, b in zip(ids, labels)
                ]
            )
        return part

    def maybe_recluster(self) -> bool:
        """Re-cluster kinds that have grown past the configured threshold.

        Returns True if any kind was (re)clustered. Meant to be called from a
        background job, not the write path.
        """
        did = False
        for kind in self._partition_kinds():
            part = self._load_partition(kind)
            if not part or part.get("centroids") is None:
                if self.count_kind(kind) > 0:
                    self._build_partition(kind)
                    did = True
                continue
            n_at_build = int(part.get("n_at_build", 0)) or 1
            growth = (self.count_kind(kind) - n_at_build) / n_at_build
            if growth >= self._storage.recluster_growth_threshold:
                self.recluster(kind=kind)
                did = True
        return did

    def recluster(
        self, kind: Optional[str] = None, sample: int = 50_000, page: int = 10_000
    ) -> Dict:
        """Full re-cluster of one kind: new centroids from a sample, then re-label
        EVERY row of that kind. With ``kind=None`` reclusters the default kind.

        Re-labelling is a stream of `topic_bucket`/`partition_version` patches
        (scalars) — vectors are never re-upserted. Queries keep using the old
        partition until `_save_partition` flips the cached centroids atomically.
        """
        kind = kind or _DEFAULT_KIND
        with telemetry.span("memory.recluster", kind=kind) as sp:
            # bumps version, saves centroids; the loop below relabels every row
            part = self._build_partition(kind=kind, sample=sample, relabel_sample=False)
            if not part:
                return {}
            cents, version = part["centroids"], int(part["version"])
            from membukkit.retrieval.buckets import assign_nearest

            relabelled = 0
            for ids, vecs in self._iter_fact_pages(page=page, kind=kind):
                if not ids:
                    continue
                labels = assign_nearest(cents, vecs)
                self._patch_rows(
                    [
                        {"id": i, "topic_bucket": int(b), "partition_version": version}
                        for i, b in zip(ids, labels)
                    ]
                )
                relabelled += len(ids)
            telemetry.set_attributes(
                sp, relabelled=relabelled, k=int(cents.shape[0]), version=version, kind=kind
            )
            telemetry.counter("membukkit.recluster").add(1, {"kind": kind})
            logger.info(
                "recluster[%s]: relabelled %d facts into %d buckets (v%d)",
                kind,
                relabelled,
                cents.shape[0],
                version,
            )
            return part

    def _iter_fact_pages(self, page: int = 10_000, kind: Optional[str] = None):
        """Yield (ids, vectors) pages over live facts (optionally one ``kind``).

        The query API has no cursor/offset, so we keyset-paginate: order by ``id``
        ascending and advance each page with an ``id > last_seen`` filter. This
        walks the whole bank with no dupes or skips (ids are the unique primary
        key), respecting Turbopuffer's per-query ``top_k`` ceiling.
        """
        page = min(page, _MAX_TOP_K)  # respect Turbopuffer's per-query top_k ceiling
        seen = 0
        last_id: Optional[str] = None
        while True:
            clauses = [["superseded_by", "Eq", ""]]
            if kind:
                clauses.append(["kind", "Eq", kind])
            if last_id is not None:
                clauses.append(["id", "Gt", last_id])
            flt = clauses[0] if len(clauses) == 1 else ["And", clauses]
            try:
                res = self._q(
                    rank_by=("id", "asc"),
                    filters=flt,
                    top_k=page,
                    include_attributes=["vector"],
                )
            except Exception as e:
                logger.warning("iter_fact_pages query failed after %d facts: %s", seen, e)
                return
            rows = _rows(res)
            if not rows:
                return
            ids = [_row_get(r, "id") for r in rows]
            vecs = np.vstack([np.asarray(_row_get(r, "vector"), dtype=np.float32) for r in rows])
            seen += len(ids)
            yield ids, vecs
            if len(rows) < page:  # last (partial) page
                return
            last_id = ids[-1]

    def _sample_vectors(self, n: int, kind: Optional[str] = None) -> Tuple[List[str], np.ndarray]:
        # A single query can't exceed Turbopuffer's top_k ceiling, so page (via
        # the cursor iterator) until we've gathered up to `n` sample vectors.
        ids: List[str] = []
        chunks: List[np.ndarray] = []
        for pids, pvecs in self._iter_fact_pages(page=min(n, _MAX_TOP_K), kind=kind):
            ids.extend(pids)
            chunks.append(pvecs)
            if len(ids) >= n:
                break
        if not chunks:
            return [], np.zeros((0, self._dim or 768), np.float32)
        vecs = np.vstack(chunks)
        return ids[:n], vecs[:n]

    def topic_exemplars(self, bucket: int, n: int = 5) -> List[str]:
        try:
            res = self._q(
                rank_by=("ts", "desc"),
                filters=["And", [["topic_bucket", "Eq", int(bucket)], ["superseded_by", "Eq", ""]]],
                top_k=n,
                include_attributes=["text"],
            )
            return [(_row_get(r, "text") or "")[:150] for r in _rows(res)]
        except Exception as e:
            logger.debug("topic_exemplars failed: %s", e)
            return []

    def delete_facts(self, fact_ids: Sequence[str]) -> int:
        """Erase specific rows, then repair supersession pointers into them.

        A fact superseded by a deleted fact is made current again, matching the
        in-memory backend: otherwise deleting a bad correction would leave the
        namespace with a dangling `superseded_by` and no current value, since
        fact queries filter on `superseded_by = ''`.
        """
        ids = [f for f in dict.fromkeys(fact_ids) if f]
        if not ids:
            return 0
        with telemetry.span("memory.delete_facts", n=len(ids)):
            orphaned: List[str] = []
            for j in range(0, len(ids), _UPSERT_BATCH):
                batch = ids[j : j + _UPSERT_BATCH]
                try:
                    res = self._q(
                        filters=["superseded_by", "In", batch],
                        include_attributes=["id"],
                        top_k=_UPSERT_BATCH,
                    )
                    orphaned.extend(_row_get(r, "id") for r in _rows(res))
                except Exception as e:  # a query failure must not skip the delete
                    logger.warning("supersession repair lookup failed: %s", e)
                self._write(deletes=batch)
            if orphaned:
                self._patch_rows(
                    [{"id": i, "superseded_by": "", "valid_to": None} for i in orphaned if i]
                )
            self._count_cache = None
        return len(ids)

    def delete(self) -> None:
        try:
            self._ns.delete_all()
        except Exception:
            try:
                self._write(delete_by_filter=["id", "Glob", "*"])
            except Exception as e:
                logger.warning("namespace delete failed: %s", e)
        self.clear()


# --------------------------------------------------------------- row helpers
def _rows(res) -> list:
    if res is None:
        return []
    for attr in ("rows", "results", "data"):
        v = getattr(res, attr, None)
        if v is not None:
            return list(v)
    if isinstance(res, dict):
        return list(res.get("rows", res.get("results", [])))
    if isinstance(res, (list, tuple)):
        return list(res)
    return []


def _multi_rows(res) -> list:
    """Rows from a multi_query response: `results[*].rows`. With server-side RRF the
    fused output is a single `Result`, so the first result's rows are the pool."""
    results = getattr(res, "results", None)
    if not results and isinstance(res, dict):
        results = res.get("results")
    if not results:
        return []
    rows = getattr(results[0], "rows", None)
    if rows is None and isinstance(results[0], dict):
        rows = results[0].get("rows")
    return list(rows or [])


def _row_get(row, key: str):
    if isinstance(row, dict):
        return row.get(key)
    # Turbopuffer rows are pydantic models with extra="allow": included attributes
    # and "$dist" live in model_extra (and "$dist" isn't a valid attribute name).
    me = getattr(row, "model_extra", None)
    if isinstance(me, dict) and key in me:
        return me[key]
    return getattr(row, key, None)


def _similarity(row) -> float:
    """cosine_distance in [0,2] -> similarity ~ 1 - dist (only for ranking)."""
    d = _row_get(row, "$dist")
    if d is None:
        d = _row_get(row, "dist")
    return float(1.0 - d) if d is not None else 0.0


def _perf(res) -> Dict:
    p = getattr(res, "performance", None)
    if isinstance(res, (list, tuple)) or p is None:
        return {}
    if isinstance(p, dict):
        return p
    if hasattr(p, "model_dump"):
        try:
            return {k: v for k, v in p.model_dump().items() if v is not None}
        except Exception:
            pass
    return {
        k: getattr(p, k) for k in dir(p) if not k.startswith("_") and not callable(getattr(p, k))
    }


def _agg_get(res, key: str):
    aggs = _row_get(res, "aggregations")
    if isinstance(aggs, dict):
        return aggs.get(key, 0)
    return _row_get(aggs, key) or 0


def _parse_iso(s: str):
    return parse_datetime(s)


def _encode_partition(part: Dict) -> str:
    buf = io.BytesIO()
    np.save(buf, part["centroids"].astype(np.float32))
    meta = {
        "centroids_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "k_eff": int(part.get("k_eff", 0)),
        "version": int(part.get("version", 0)),
        "sizes": {str(k): int(v) for k, v in part.get("sizes", {}).items()},
        "n_at_build": int(part.get("n_at_build", 0)),
    }
    return json.dumps(meta)


def _decode_partition(payload: str) -> Optional[Dict]:
    if not payload:
        return None
    meta = json.loads(payload)
    buf = io.BytesIO(base64.b64decode(meta["centroids_b64"]))
    cents = np.load(buf)
    return {
        "centroids": np.asarray(cents, dtype=np.float32),
        "k_eff": int(meta.get("k_eff", cents.shape[0])),
        "version": int(meta.get("version", 0)),
        "sizes": {int(k): int(v) for k, v in meta.get("sizes", {}).items()},
        "n_at_build": int(meta.get("n_at_build", 0)),
    }
