"""Tests for bucket transparency & control: topic-scoped exclusion, raw-cosine
traces, per-lane trace breakdown, and the eval's gold->bucket mapping.

The exclusion hook is the control interface of the evidence gate: closing a
bucket must make its facts unreachable (no fallback leak), while the trace must
expose which buckets were opened, with what route probability and raw centroid
cosine, per union lane.
"""

from __future__ import annotations

import numpy as np

from membukkit.config import PromptConfig, RetrievalConfig
from membukkit.pipeline import MemorySystem
from membukkit.retrieval.buckets import build_topic_partition, route_topic
from test_union_parity import FakeDistiller, FakeEncoder, FakeReranker


# --------------------------------------------------------------------- routing


def _partition(n=30, dim=16, seed=0, k=4):
    rng = np.random.default_rng(seed)
    embs = rng.standard_normal((n, dim)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return build_topic_partition(embs, k=k), embs


def test_route_topic_trace_has_raw_cosine():
    part, embs = _partition()
    _, trace = route_topic(part, embs[0], budget=0.3, record=False)
    assert trace["buckets"], "expected at least one opened bucket"
    for entry in trace["buckets"]:
        assert "cos" in entry and -1.0 <= entry["cos"] <= 1.0
        assert "route_prob" in entry


def test_route_topic_exclude_closes_bucket():
    part, embs = _partition()
    cand, trace = route_topic(part, embs[0], budget=0.3, record=False)
    top_bucket = trace["buckets"][0]["bucket"]

    cand_ex, trace_ex = route_topic(part, embs[0], budget=0.3, record=False, exclude=[top_bucket])
    opened = {b["bucket"] for b in trace_ex["buckets"]}
    assert top_bucket not in opened
    blocked_rows = set(part["by_bucket"][top_bucket])
    assert not blocked_rows & set(cand_ex), "excluded bucket's facts leaked"
    assert trace_ex["excluded_buckets"] == [top_bucket]


def test_route_topic_exclude_all_returns_empty_not_full_scan():
    part, embs = _partition()
    all_buckets = list(part["by_bucket"].keys())
    cand, trace = route_topic(part, embs[0], budget=0.3, record=False, exclude=all_buckets)
    assert cand == []
    assert trace["n_scanned"] == 0


def test_route_topic_single_bucket_excluded():
    part, embs = _partition(n=4)  # n<6 -> k_eff==1
    assert part["k_eff"] == 1
    cand, trace = route_topic(part, embs[0], budget=0.3, record=False, exclude=[0])
    assert cand == [] and trace["buckets"] == []


# ------------------------------------------------------------------- pipeline


def _mem(union=True):
    cfg = RetrievalConfig(
        union=union,
        bucket_mode="topic",
        num_buckets=4,
        scan_budget=0.3,
        top_k=5,
        rerank_cap=50,
    )
    return MemorySystem(
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        llm_fn=lambda p: "",
        retrieval=cfg,
        prompts=PromptConfig.default(),
        distiller=FakeDistiller(),
    )


def _sessions(n_sessions=6, turns_per=4):
    topics = ["hiking gear", "sourdough baking", "jazz piano", "marathon training"]
    sessions = []
    for s in range(n_sessions):
        topic = topics[s % len(topics)]
        sess = []
        for t in range(turns_per):
            sess.append(
                {"role": "user", "content": f"session{s} turn{t} about {topic} detail{s}{t}"}
            )
        sessions.append(sess)
    dates = [f"2023/0{(s % 9) + 1}/01 (Mon) 10:00" for s in range(n_sessions)]
    return sessions, dates


def test_answer_trace_has_lane_breakdown():
    mem = _mem()
    sessions, dates = _sessions()
    mem.ingest(sessions=sessions, dates=dates)
    res = mem.answer("what about hiking gear?", generate_answer=False)
    assert set(res.trace.lanes) == {"verbatim", "atomic"}
    for lane in res.trace.lanes.values():
        assert lane["buckets"], "lane should have opened buckets"
        assert all("cos" in b for b in lane["buckets"])


def test_answer_exclude_buckets_removes_lane_evidence():
    mem = _mem()
    sessions, dates = _sessions()
    mem.ingest(sessions=sessions, dates=dates)

    q = "what about hiking gear?"
    base = mem.answer(q, generate_answer=False)
    v_opened = [b["bucket"] for b in base.trace.lanes["verbatim"]["buckets"]]
    a_opened = [b["bucket"] for b in base.trace.lanes["atomic"]["buckets"]]

    blocked = {"verbatim": v_opened, "atomic": a_opened}
    res = mem.answer(q, generate_answer=False, exclude_buckets=blocked)
    for lane_name, blocked_ids in blocked.items():
        lane = res.trace.lanes[lane_name]
        assert lane.get("excluded_buckets") == sorted(set(blocked_ids))
        reopened = {b["bucket"] for b in lane["buckets"]} & set(blocked_ids)
        assert not reopened, f"{lane_name}: excluded buckets reopened"
    # Retrieval must differ once its top buckets are closed.
    assert res.facts != base.facts


def test_lane_view_labels_match_routing_partition():
    mem = _mem()
    sessions, dates = _sessions()
    mem.ingest(sessions=sessions, dates=dates)
    backend = mem._backend
    for kind in ("verbatim", "atomic"):
        view = backend.lane_view(kind)
        part = backend._kind_partition(kind)
        assert view["k_eff"] == part["k_eff"]
        assert view["labels"] == [int(x) for x in part["labels"]]
        assert len(view["ids"]) == len(view["labels"]) == len(view["sources"])
        # sources carry the ingest session backpointer used for gold mapping
        assert all(s.startswith("ingest:") for s in view["sources"])


# ------------------------------------------------------------------ eval glue


def test_gold_bucket_map_and_blocking_arms():
    from membukkit.cli.eval_legacy import _blocked_buckets, _gold_bucket_map, _lane_views
    from membukkit.data.instance import LongMemEvalInstance

    sessions, dates = _sessions()
    mem = _mem()
    mem.ingest(sessions=sessions, dates=dates)

    # Real LongMemEval instance shape, with session 0 as the gold session.
    inst = LongMemEvalInstance(
        {
            "question_id": "q1",
            "question": "what about hiking gear?",
            "question_type": "multi-session",
            "answer": "gt",
            "haystack_sessions": sessions,
            "haystack_dates": dates,
            "haystack_session_ids": [f"s{i}" for i in range(len(sessions))],
            "answer_session_ids": ["s0"],
        }
    )

    views = _lane_views(mem._backend)
    assert set(views) == {"verbatim", "atomic"}

    gold = _gold_bucket_map(inst, views)
    assert gold["verbatim"], "gold turns must map to at least one verbatim bucket"
    assert gold["atomic"], "gold session's atomic facts must map to buckets"
    # Every gold turn's bucket must be within the lane's bucket range.
    for lane, ids in gold.items():
        k_eff = views[lane]["k_eff"]
        assert all(0 <= b < k_eff for b in ids)

    blocked_gold = _blocked_buckets(gold, "gold", views, "seed:q1")
    assert blocked_gold == {k: v for k, v in gold.items() if v}

    blocked_rand = _blocked_buckets(gold, "random", views, "seed:q1")
    assert blocked_rand is not None
    for lane, ids in blocked_rand.items():
        assert not set(ids) & set(gold[lane]), "random arm must avoid gold buckets"
        assert len(ids) <= len(gold[lane])
    # Deterministic under the same seed key, different under another.
    assert blocked_rand == _blocked_buckets(gold, "random", views, "seed:q1")

    assert _blocked_buckets(gold, "none", views, "seed:q1") is None
