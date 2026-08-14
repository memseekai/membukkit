"""Mocked tests for TurbopufferBackend — verify request shapes & parsing.

No live Turbopuffer: a FakeNamespace records calls and returns canned rows. The
SDK is never imported (we inject `_namespace` directly).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from membukkit.config import RetrievalConfig, StorageConfig
from membukkit.storage.base import FactRecord
from membukkit.storage.turbopuffer import TurbopufferBackend


class FakeEncoder:
    def __init__(self, dim=16):
        self.dim = dim

    def encode(self, texts, normalize=True, show_progress=False):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        out = []
        for t in items:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(self.dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-8
            out.append(v)
        arr = np.vstack(out).astype(np.float32)
        return arr[0] if single else arr


class FakeResult:
    def __init__(self, rows=None, aggregations=None):
        self.rows = rows or []
        self.aggregations = aggregations or {}
        self.performance = {"cache_temperature": "warm", "server_total_ms": 7}


class FakeMultiResult:
    def __init__(self, results):
        self.results = results
        self.performance = {"cache_temperature": "warm", "server_total_ms": 7}


class FakeNamespace:
    def __init__(self, search_rows=None, existing=None):
        self.calls = []
        self.writes = []
        self._search_rows = search_rows or []
        self._existing = existing or []

    def write(self, **kwargs):
        self.writes.append(kwargs)
        return FakeResult()

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        flt = kwargs.get("filters")
        if kwargs.get("aggregate_by"):
            return FakeResult(aggregations={"n": 5})
        if flt and flt[0] == "id" and flt[1] == "In":
            return FakeResult(rows=[{"id": i} for i in self._existing])
        if flt and flt[0] == "id" and flt[1] == "Eq":
            return FakeResult(rows=[])
        return FakeResult(rows=self._search_rows)

    def multi_query(self, queries=None, rerank_by=None):
        self.calls.append(("multi_query", {"queries": queries, "rerank_by": rerank_by}))
        # Mirror the real shape: results[*].rows (RRF fuses into a single Result).
        return FakeMultiResult(results=[FakeResult(rows=self._search_rows)])


class FilteringFakeNamespace:
    """A fake that STORES rows and interprets the real filter grammar.

    `FakeNamespace` returns canned rows regardless of filters, so filter-shape
    bugs (unreachable buckets, date-range exclusions, wrong counts) pass it
    silently. This fake evaluates And/Or/Eq/In/ContainsAny/Gt/Gte/Lt/Glob
    against stored rows and brute-forces ANN/BM25 ranking, so those bugs fail.
    """

    def __init__(self):
        self.rows = {}
        self.calls = []
        self.writes = []

    def write(self, **kwargs):
        self.writes.append(kwargs)
        for r in kwargs.get("upsert_rows") or []:
            self.rows[r["id"]] = dict(r)
        for p in kwargs.get("patch_rows") or []:
            self.rows.setdefault(p["id"], {"id": p["id"]}).update(p)
        if kwargs.get("delete_by_filter") is not None:
            self.rows = {}
        return FakeResult()

    def delete_all(self):
        self.rows = {}

    @staticmethod
    def _norm(v):
        if isinstance(v, datetime) and v.tzinfo is not None:
            return v.astimezone(timezone.utc).replace(tzinfo=None)
        return v

    def _matches(self, row, flt):
        if flt is None:
            return True
        if flt[0] == "And":
            return all(self._matches(row, c) for c in flt[1])
        if flt[0] == "Or":
            return any(self._matches(row, c) for c in flt[1])
        fld, op, val = flt
        have = self._norm(row.get(fld))
        val = self._norm(val)
        if op == "Eq":
            return have == val
        if op == "In":
            return have in val
        if op == "ContainsAny":
            return bool(set(have or []) & set(val))
        if op in ("Gt", "Gte", "Lt"):
            if have is None:
                return False
            return {"Gt": have > val, "Gte": have >= val, "Lt": have < val}[op]
        if op == "Glob":
            import fnmatch

            return fnmatch.fnmatch(str(have or ""), val)
        raise AssertionError(f"unsupported filter op: {op}")

    def _run(self, **kwargs):
        matches = [r for r in self.rows.values() if self._matches(r, kwargs.get("filters"))]
        if kwargs.get("aggregate_by"):
            return FakeResult(aggregations={"n": len(matches)})
        rank = kwargs.get("rank_by")
        if rank and rank[0] == "vector":
            qv = np.asarray(rank[2], dtype=np.float32)
            scored = [
                (1.0 - float(np.asarray(r["vector"], np.float32) @ qv), r)
                for r in matches
                if r.get("vector") is not None
            ]
            scored.sort(key=lambda x: x[0])
            out = [{**r, "$dist": d} for d, r in scored]
        elif rank and rank[0] == "text":  # crude BM25: query-token overlap
            q_tokens = set(str(rank[2]).lower().split())
            scored = [
                (len(q_tokens & set((r.get("text") or "").lower().split())), r) for r in matches
            ]
            out = [dict(r) for s, r in sorted(scored, key=lambda x: -x[0]) if s > 0]
        elif rank and rank[0] == "ts":
            out = sorted(
                (dict(r) for r in matches if r.get("ts") is not None),
                key=lambda r: self._norm(r["ts"]),
                reverse=rank[1] == "desc",
            )
        else:  # ("id", "asc")
            out = sorted((dict(r) for r in matches), key=lambda r: r["id"])
        return FakeResult(rows=out[: kwargs.get("top_k") or len(out)])

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        return self._run(**kwargs)

    def multi_query(self, queries: list[dict] | None = None, rerank_by=None):
        self.calls.append(("multi_query", {"queries": queries, "rerank_by": rerank_by}))
        # RRF-fuse the lanes, mirroring the server's single fused Result.
        scores, by_id = {}, {}
        for q in queries or ():
            for rank, r in enumerate(self._run(**q).rows):
                scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (60 + rank)
                by_id[r["id"]] = r
        fused = [by_id[i] for i in sorted(scores, key=lambda i: scores[i], reverse=True)]
        return FakeMultiResult(results=[FakeResult(rows=fused)])


def _backend(cfg=None, ns=None):
    be = TurbopufferBackend(
        cfg or RetrievalConfig(),
        FakeEncoder(),
        StorageConfig(backend="turbopuffer", namespace="mem_test", region="x"),
    )
    be._namespace = ns
    return be


def test_upsert_builds_rows_and_skips_existing():
    ns = FakeNamespace(existing=[])
    be = _backend(ns=ns)
    n = be.upsert_facts(
        [
            FactRecord(text="alice likes tea", timestamp=datetime(2024, 6, 1), entities=["alice"]),
            FactRecord(text="bob likes coffee", timestamp=datetime(2024, 6, 2)),
        ]
    )
    assert n == 2
    upserts = [w for w in ns.writes if "upsert_rows" in w]
    assert upserts, "expected an upsert write"
    rows = upserts[0]["upsert_rows"]
    assert upserts[0]["distance_metric"] == "cosine_distance"
    r0 = rows[0]
    assert set(
        [
            "id",
            "vector",
            "text",
            "ts",
            "kind",
            "topic_bucket",
            "entities",
            "superseded_by",
            "partition_version",
        ]
    ).issubset(r0.keys())
    assert r0["superseded_by"] == ""
    assert r0["kind"] == "atomic"  # default kind is stored on the row
    assert len(r0["vector"]) == 16


def test_upsert_dedups_against_existing():
    seed = FactRecord(text="alice likes tea")
    ns = FakeNamespace(existing=[seed.ensure_id()])
    be = _backend(ns=ns)
    n = be.upsert_facts([FactRecord(text="alice likes tea")])
    assert n == 0
    assert not [w for w in ns.writes if "upsert_rows" in w]


def _canned_rows():
    return [
        {
            "id": "a",
            "text": "alice likes tea",
            "ts": "2024-06-01T10:30:00Z",
            "topic_bucket": 0,
            "entities": ["alice"],
            "time_bucket": "2024-06",
            "$dist": 0.1,
        },
        {
            "id": "b",
            "text": "bob likes coffee",
            "ts": "2024-06-02T09:00:00-05:00",
            "topic_bucket": 1,
            "entities": ["bob"],
            "time_bucket": "2024-06",
            "$dist": 0.3,
        },
    ]


def test_candidates_gated_one_multiquery_with_bucket_filter():
    cfg = RetrievalConfig(retrieval_mode="gated", bm25_lane=True, pool_size=8)
    ns = FakeNamespace(search_rows=_canned_rows())
    be = _backend(cfg=cfg, ns=ns)
    # Seed the (atomic) partition + count so no extra round trips happen.
    be._partitions = {
        "atomic": {
            "centroids": np.eye(4, 16, dtype=np.float32),
            "k_eff": 4,
            "version": 1,
            "sizes": {0: 10, 1: 8, 2: 5, 3: 2},
        }
    }
    be._count_cache = 25

    pool = be.candidates("what does alice like", top_k=10)

    mq = [c for c in ns.calls if c[0] == "multi_query"]
    assert len(mq) == 1, "exactly one search round trip"
    queries = mq[0][1]["queries"]
    assert len(queries) == 2  # dense + BM25 lanes
    assert mq[0][1]["rerank_by"] == ("RRF",)
    # gated mode must constrain by topic_bucket somewhere in the filter
    assert "topic_bucket" in str(queries[0].get("filters"))
    assert len(pool.candidates) == 2
    assert pool.candidates[0].text == "alice likes tea"
    assert pool.candidates[0].timestamp.tzinfo is timezone.utc
    assert pool.has_cosine is True
    assert pool.trace["mode"] == "gated"
    assert pool.trace["perf"]["cache_temperature"] == "warm"


def test_candidates_open_mode_no_bucket_filter():
    cfg = RetrievalConfig(retrieval_mode="open", bm25_lane=False, pool_size=8)
    ns = FakeNamespace(search_rows=_canned_rows())
    be = _backend(cfg=cfg, ns=ns)
    be._partitions = {
        "atomic": {
            "centroids": np.eye(4, 16, dtype=np.float32),
            "k_eff": 4,
            "version": 1,
            "sizes": {0: 10, 1: 8},
        }
    }
    be._count_cache = 25

    pool = be.candidates("what does alice like", top_k=10)
    # bm25 off -> single query, not multi_query
    q = [c for c in ns.calls if c[0] == "query"]
    assert any("vector" in str(c[1].get("rank_by")) for c in q)
    assert all(c[0] != "multi_query" for c in ns.calls)
    # open mode: filter must NOT pin topic_bucket
    vq = [c for c in q if "vector" in str(c[1].get("rank_by"))][0]
    assert "topic_bucket" not in str(vq[1].get("filters"))
    assert pool.trace["mode"] == "open"


def test_supersede_patches_scalars_only():
    ns = FakeNamespace()
    be = _backend(ns=ns)
    be.supersede([("old1", "new1"), ("old2", "new2")])
    patches = [w for w in ns.writes if "patch_rows" in w]
    assert patches
    rows = patches[0]["patch_rows"]
    assert {r["id"] for r in rows} == {"old1", "old2"}
    assert all(r["superseded_by"].startswith("new") for r in rows)
    assert all(r["tag"] == "UPDATED" for r in rows)
    assert all(r["valid_to"].tzinfo is timezone.utc for r in rows)
    # crucially, no vector in a supersede patch
    assert all("vector" not in r for r in rows)


def test_recluster_relabels_via_patches():
    rows = [
        {"id": f"f{i}", "vector": list(np.random.default_rng(i).standard_normal(16))}
        for i in range(6)
    ]

    class ReclusterNS(FakeNamespace):
        def query(self, **kwargs):
            self.calls.append(("query", kwargs))
            if kwargs.get("aggregate_by"):
                return FakeResult(aggregations={"n": 6})
            inc = kwargs.get("include_attributes") or []
            inc = list(inc) if not isinstance(inc, bool) else []
            # vector pulls (sample + each iter page) return the 6 facts; the
            # iterator stops itself because 6 < page and there is no cursor. The
            # kind-scoped filter is now an And([live, kind, ...]) clause.
            if "vector" in inc:
                return FakeResult(rows=rows)
            return FakeResult(rows=[])  # no existing partition meta

    ns = ReclusterNS()
    be = _backend(cfg=RetrievalConfig(num_buckets=3), ns=ns)
    part = be.recluster(sample=100, page=10)
    assert part["centroids"].shape[0] >= 1
    # every fact got a topic_bucket + partition_version patch, no vectors
    patches = [w for w in ns.writes if "patch_rows" in w]
    patched_ids = {r["id"] for w in patches for r in w["patch_rows"]}
    assert patched_ids == {f"f{i}" for i in range(6)}
    assert all(
        "topic_bucket" in r and "partition_version" in r and "vector" not in r
        for w in patches
        for r in w["patch_rows"]
    )


def test_partition_encode_roundtrip():
    from membukkit.storage.turbopuffer import _encode_partition, _decode_partition

    part = {
        "centroids": np.random.default_rng(0).standard_normal((4, 16)).astype(np.float32),
        "k_eff": 4,
        "version": 3,
        "sizes": {0: 2, 1: 3},
    }
    dec = _decode_partition(_encode_partition(part))
    assert dec is not None
    assert dec["k_eff"] == 4 and dec["version"] == 3
    assert dec["sizes"] == {0: 2, 1: 3}
    assert np.allclose(dec["centroids"], part["centroids"])


# --------------------------------------------------------- recall regressions
def test_pre_partition_facts_relabelled_and_retrievable_after_warm():
    """Facts ingested before the first partition (bucket -1) must become
    routable once partition()/warm runs — not only after a full recluster."""
    cfg = RetrievalConfig(retrieval_mode="gated", bm25_lane=False, pool_size=8, num_buckets=2)
    ns = FilteringFakeNamespace()
    be = _backend(cfg=cfg, ns=ns)
    be.upsert_facts(
        [
            FactRecord(text=f"note {i} about hobby {i % 3}", timestamp=datetime(2024, 6, 1 + i))
            for i in range(6)
        ]
    )
    fact_rows = [r for r in ns.rows.values() if not r["id"].startswith("__partition__")]
    assert all(r["topic_bucket"] == -1 for r in fact_rows), "cold writes are unlabelled"

    be.partition()  # builds centroids AND relabels the sampled rows
    fact_rows = [r for r in ns.rows.values() if not r["id"].startswith("__partition__")]
    assert all(r["topic_bucket"] >= 0 for r in fact_rows)

    pool = be.candidates("hobby", top_k=10)
    assert pool.candidates, "post-warm gated retrieval must reach the facts"


def test_unlabelled_bucket_minus_one_always_reachable_in_gated_mode():
    """Even without a relabel, bucket -1 rides along with every opened set."""
    cfg = RetrievalConfig(retrieval_mode="gated", bm25_lane=False, pool_size=8)
    ns = FilteringFakeNamespace()
    be = _backend(cfg=cfg, ns=ns)
    be.upsert_facts(
        [FactRecord(text=f"memo {i}", timestamp=datetime(2024, 6, 1)) for i in range(4)]
    )
    # Partition exists (e.g. built by another worker) but these rows were never
    # relabelled — they still carry topic_bucket=-1.
    be._partitions = {
        "atomic": {
            "centroids": np.eye(2, 16, dtype=np.float32),
            "k_eff": 2,
            "version": 1,
            "sizes": {0: 2, 1: 2},
        }
    }
    pool = be.candidates("memo", top_k=10)
    assert pool.candidates, "unlabelled (-1) facts must never be filtered out"


def test_undated_fact_retrievable_in_temporal_query():
    cfg = RetrievalConfig(retrieval_mode="open", bm25_lane=False, pool_size=8)
    ns = FilteringFakeNamespace()
    be = _backend(cfg=cfg, ns=ns)
    be.upsert_facts(
        [
            FactRecord(text="alice moved to berlin"),  # no timestamp
            FactRecord(text="bob visited paris", timestamp=datetime(2024, 6, 1)),
        ]
    )
    pool = be.candidates("what happened in 2024?", top_k=10, is_temporal=True)
    texts = {c.text for c in pool.candidates}
    assert "alice moved to berlin" in texts, "undated facts must survive date filters"
    undated = next(c for c in pool.candidates if c.text == "alice moved to berlin")
    assert undated.timestamp is None, "TS_UNKNOWN sentinel must not leak as 1970"


def test_count_correct_on_warm_namespace_new_backend_instance():
    """A fresh backend over a pre-populated namespace (new worker / LRU
    re-create) must not seed its count cache from zero."""
    ns = FilteringFakeNamespace()
    be = _backend(ns=ns)
    be.upsert_facts([FactRecord(text=f"old fact {i}") for i in range(5)])

    be2 = _backend(ns=ns)  # fresh instance, same namespace
    be2.upsert_facts([FactRecord(text="brand new fact")])
    assert be2.count() == 6


def test_count_failure_is_not_cached_as_zero():
    class FailOnceAggNS(FilteringFakeNamespace):
        def __init__(self):
            super().__init__()
            self.failed = False

        def query(self, **kwargs):
            if kwargs.get("aggregate_by") and not self.failed:
                self.failed = True
                raise RuntimeError("transient aggregate failure")
            return super().query(**kwargs)

    ns = FailOnceAggNS()
    ns.write(upsert_rows=[{"id": "x", "text": "t", "superseded_by": ""}])
    be = _backend(ns=ns)
    assert be.count() == 0  # transient failure surfaces as 0 for this call...
    assert be.count() == 1  # ...but heals on the next call instead of sticking


def test_per_kind_partitions_and_lane_isolation():
    """Union backend builds a SEPARATE partition per kind and each lane's
    retrieval returns only its own kind's rows."""
    cfg = RetrievalConfig(
        retrieval_mode="gated", bm25_lane=False, pool_size=8, num_buckets=2, union=True
    )
    ns = FilteringFakeNamespace()
    be = _backend(cfg=cfg, ns=ns)
    be.upsert_facts(
        [FactRecord(text=f"raw turn {i} about hobby {i % 2}", kind="verbatim") for i in range(5)]
        + [FactRecord(text=f"atomic fact {i} about hobby {i % 2}", kind="atomic") for i in range(5)]
    )
    assert be.count_kind("verbatim") == 5
    assert be.count_kind("atomic") == 5

    # A background recluster builds one partition per kind.
    assert be.maybe_recluster() is True
    assert "__partition__:verbatim" in ns.rows
    assert "__partition__:atomic" in ns.rows
    # Each partition doc is tagged with its own kind.
    assert ns.rows["__partition__:verbatim"]["kind"] == "verbatim"
    assert ns.rows["__partition__:atomic"]["kind"] == "atomic"

    v = be.candidates("hobby", top_k=10, kind="verbatim")
    a = be.candidates("hobby", top_k=10, kind="atomic")
    assert v.candidates and a.candidates
    assert all(c.kind == "verbatim" for c in v.candidates), "verbatim lane leaked atomic rows"
    assert all(c.kind == "atomic" for c in a.candidates), "atomic lane leaked verbatim rows"
    # The lane trace is scoped to that kind's fact count, not the global total.
    assert v.trace["n_facts"] == 5
    assert a.trace["n_facts"] == 5


def test_memorysystem_union_over_turbopuffer_merges_lane_perf():
    """End-to-end: MemorySystem union over the Turbopuffer backend retrieves both
    lanes and reports the summed (per-kind) server cost/latency in the trace."""
    from membukkit.config import PromptConfig
    from membukkit.pipeline import MemorySystem

    class Reranker:
        def score(self, query, texts, batch_size=64):
            qs = set(query.lower().split())
            return np.asarray(
                [len(qs & set(t.lower().split())) + (abs(hash(t)) % 1000) / 1e6 for t in texts],
                dtype=np.float32,
            )

    class Distiller:
        subject = None

        def distill(self, key, transcript, date):
            tag = abs(hash(transcript)) % 100000
            return [
                (i, f"atom{tag}_{i} {ln.strip()[:40]}")
                for i, ln in enumerate(transcript.split("\n"))
                if ln.strip()
            ]

        def save(self):
            pass

    cfg = RetrievalConfig(
        retrieval_mode="gated", bm25_lane=False, pool_size=8, num_buckets=2, union=True, top_k=5
    )
    ns = FilteringFakeNamespace()
    be = _backend(cfg=cfg, ns=ns)
    mem = MemorySystem(
        encoder=FakeEncoder(),
        reranker=Reranker(),
        llm_fn=lambda p: "unused",
        retrieval=cfg,
        prompts=PromptConfig.default(),
        distiller=Distiller(),
        backend=be,
    )
    mem.ingest(
        sessions=[
            [
                {"role": "user", "content": "I adopted a dog named Luna"},
                {"role": "user", "content": "Luna is a golden retriever"},
            ]
        ],
        dates=["2024-06-01"],
    )
    be.maybe_recluster()

    assert be.count_kind("verbatim") > 0 and be.count_kind("atomic") > 0
    res = mem.answer("what dog is Luna", generate_answer=False)
    assert res.facts, "union retrieval must return facts"

    perf = res.trace.perf
    assert "per_lane" in perf, "union trace must keep a per-lane cost breakdown"
    assert perf["server_total_ms"] == (
        perf["per_lane"]["verbatim"]["server_ms"] + perf["per_lane"]["atomic"]["server_ms"]
    )


def test_iso_query_time_range_builds_day_filter():
    from membukkit.retrieval.query_filters import build_filter, query_time_range

    rng = query_time_range("what happened on 2024-06-01T10:30:00Z?")
    assert rng is not None
    start, end = rng
    assert start.isoformat() == "2024-06-01T00:00:00+00:00"
    assert end.isoformat() == "2024-06-02T00:00:00+00:00"

    flt = build_filter(time_range=rng, live_only=True)
    assert "Gte" in str(flt)
    assert "Lt" in str(flt)
