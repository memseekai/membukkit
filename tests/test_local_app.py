"""Local GUI API tests: upload -> facts -> provenance -> partition -> ask.

Runs offline: fake encoder/reranker/LLM injected via membukkit.cli.common.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from membukkit.config import PromptConfig, RetrievalConfig  # noqa: E402
from membukkit.pipeline import MemorySystem  # noqa: E402


class FakeEncoder:
    dim = 16

    def encode(self, texts, normalize=True, show_progress=False):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vecs = []
        for t in items:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(self.dim).astype(np.float32)
            vecs.append(v / np.linalg.norm(v))
        out = np.stack(vecs)
        return out[0] if single else out


class FakeReranker:
    def score(self, query, texts):
        return np.zeros(len(texts), dtype=np.float32)


def fake_llm(prompt: str) -> str:
    return "a synthetic answer"


class FakeDistiller:
    """Distills every session into two dated atomic facts."""

    subject = None

    def distill(self, key, transcript, date_str, mode="chat"):
        return [
            (0, f"The quarterly report deadline was set to April 5th (noted {date_str})."),
            (1, f"Finance owned the revenue section (noted {date_str})."),
        ]


def _make_client(tmp_path, monkeypatch, distiller=None, llm_fn=fake_llm):
    monkeypatch.setenv("MEMBUKKIT_HOME", str(tmp_path))

    def fake_build_system(
        llm="x", encoder_spec="y", distill=True, retrieval=None, prompts=None
    ):
        return MemorySystem(
            encoder=FakeEncoder(),
            reranker=FakeReranker(),
            llm_fn=llm_fn,
            retrieval=retrieval or RetrievalConfig(),
            prompts=prompts or PromptConfig.default(),
            distiller=distiller,
        )

    import membukkit.cli.common as common

    monkeypatch.setattr(common, "build_system", fake_build_system)

    from membukkit.service.local_app import create_local_app

    return TestClient(create_local_app())


@pytest.fixture()
def client(tmp_path, monkeypatch):
    return _make_client(tmp_path, monkeypatch)


CHAT = b"""[
  {"date": "2024-03-01", "turns": [
    {"role": "user", "content": "The quarterly report is due April 5th."},
    {"role": "user", "content": "Finance owns the revenue section this time."}
  ]}
]"""


def test_full_flow(client):
    # empty state
    assert client.get("/api/stores").json()["stores"] == []

    # upload a chat export
    r = client.post(
        "/api/stores/work/upload",
        files=[("files", ("notes.json", CHAT, "application/json"))],
    )
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["new_facts"] == 2
    assert body["n_facts"] == 2

    # store listing and overview
    stores = client.get("/api/stores").json()["stores"]
    assert stores[0]["name"] == "work" and stores[0]["n_facts"] == 2
    ov = client.get("/api/stores/work/overview").json()
    assert ov["n_verbatim"] == 2 and len(ov["documents"]) == 1
    assert ov.get("latest_fact_date") == "2024-03-01"
    assert "suggested_questions" in ov
    assert isinstance(ov["suggested_questions"], list)

    # persisted chips surface on overview (generation is tested separately)
    from membukkit.storage.localstore import LocalStore

    LocalStore("work").update_meta(
        suggested_questions=["When is the quarterly report due?"]
    )
    ov2 = client.get("/api/stores/work/overview").json()
    assert ov2["suggested_questions"] == ["When is the quarterly report due?"]

    # facts browser with provenance fields
    facts = client.get("/api/stores/work/facts").json()
    assert facts["total"] == 2
    fact = facts["facts"][0]
    assert fact["doc_name"] == "notes.json"
    assert fact["source_ref"].startswith("session:0/turn:")

    # provenance drill-down resolves to the raw passage
    src = client.get(f"/api/stores/work/facts/{fact['id']}/source").json()
    assert src["source"]["name"] == "notes.json"
    highlighted = src["source"]["turns"][src["source"]["highlight"]]
    assert highlighted["content"] == fact["text"]

    # partition view — no atomic facts (distiller off) so verbatim is the lane
    part = client.get("/api/stores/work/partition").json()
    assert part["n_facts"] == 2 and part["k_eff"] >= 1
    assert part["lane"] == "verbatim"
    assert sum(b["size"] for b in part["buckets"]) == 2

    # ask returns answer + trace + evidence
    ans = client.post("/api/stores/work/ask", json={"question": "when is the report due?"}).json()
    assert ans["answer"] == "a synthetic answer"
    assert ans["trace"]["n_facts"] == 2
    assert ans["trace"]["est_reader_tokens"] > 0
    assert ans["evidence"] and ans["evidence"][0]["doc_name"] == "notes.json"
    assert "question_date" in ans
    assert "bucket_labels" in ans["trace"]
    assert "kind" in ans["evidence"][0]

    # as-of date is accepted (temporal control for the reader)
    ans2 = client.post(
        "/api/stores/work/ask",
        json={"question": "when is the report due?", "question_date": "2024-06-01"},
    ).json()
    assert ans2["question_date"] == "2024-06-01"

    # deletion
    assert client.delete("/api/stores/work").status_code == 200
    assert client.get("/api/stores/work/overview").status_code == 404


def test_partition_uses_atomic_lane(tmp_path, monkeypatch):
    calls = []

    def counting_llm(prompt: str) -> str:
        calls.append(prompt)
        return f"label {len(calls)}"

    client = _make_client(tmp_path, monkeypatch, distiller=FakeDistiller(), llm_fn=counting_llm)
    r = client.post(
        "/api/stores/work/upload",
        files=[("files", ("notes.json", CHAT, "application/json"))],
    )
    assert r.status_code == 200
    ov = client.get("/api/stores/work/overview").json()
    assert ov["n_atomic"] == 2 and ov["n_verbatim"] == 2

    # the map is over the atomic (distilled) lane, not the mixed global bank
    part = client.get("/api/stores/work/partition").json()
    assert part["lane"] == "atomic"
    assert part["n_facts"] == 2
    assert sum(b["size"] for b in part["buckets"]) == 2
    for b in part["buckets"]:
        assert b["exemplars"]
        for ex in b["exemplars"]:
            assert "noted" in ex  # distilled fact texts, never raw turns

    # bucket browsing filters to the same lane-local bucket ids
    facts = client.get("/api/stores/work/facts?bucket=0&kind=atomic").json()["facts"]
    assert facts and all(f["kind"] == "atomic" for f in facts)

    # labeling works on the first request and is cached afterwards
    part = client.get("/api/stores/work/partition?label=true").json()
    assert part["buckets"] and all(b["label"] for b in part["buckets"])
    n_calls = len(calls)
    part2 = client.get("/api/stores/work/partition?label=true").json()
    assert len(calls) == n_calls
    assert [b["label"] for b in part2["buckets"]] == [b["label"] for b in part["buckets"]]

    # refresh=true forces regeneration
    client.get("/api/stores/work/partition?label=true&refresh=true")
    assert len(calls) > n_calls


def test_atomic_fact_source_drilldown(tmp_path, monkeypatch):
    """Atomic facts carry turn-level provenance; the modal endpoint resolves it."""
    client = _make_client(tmp_path, monkeypatch, distiller=FakeDistiller())
    client.post(
        "/api/stores/prov/upload",
        files=[("files", ("notes.json", CHAT, "application/json"))],
    )
    atomic = client.get("/api/stores/prov/facts?kind=atomic").json()["facts"]
    assert len(atomic) == 2
    # FakeDistiller backpointers facts to turns 0 and 1 -> stored refs
    assert {f["source_ref"] for f in atomic} == {"session:0/turn:0", "session:0/turn:1"}

    for fact in atomic:
        src = client.get(f"/api/stores/prov/facts/{fact['id']}/source").json()["source"]
        assert src["highlight_kind"] == "stored"
        want = int(fact["source_ref"].rsplit(":", 1)[1])
        highlighted = src["turns"][src["highlight"]]["content"]
        assert highlighted == ["The quarterly report is due April 5th.",
                              "Finance owns the revenue section this time."][want]


class SessionOnlyDistiller:
    """Legacy behavior: no turn backpointers (idx -1 falls back to session refs)."""

    subject = None

    def distill(self, key, transcript, date_str, mode="chat"):
        return [(-1, "The quarterly report deadline is April 5th.")]


def test_legacy_session_ref_uses_lexical_fallback(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, distiller=SessionOnlyDistiller())
    client.post(
        "/api/stores/legacy/upload",
        files=[("files", ("notes.json", CHAT, "application/json"))],
    )
    fact = client.get("/api/stores/legacy/facts?kind=atomic").json()["facts"][0]
    assert fact["source_ref"] == "session:0"  # no turn info stored

    src = client.get(f"/api/stores/legacy/facts/{fact['id']}/source").json()["source"]
    assert src["highlight_kind"] == "lexical"
    assert src["turns"][src["highlight"]]["content"] == "The quarterly report is due April 5th."


def test_force_distill_rescues_verbatim_only_store(tmp_path, monkeypatch):
    """A store ingested without distillation gains atomic facts via /distill."""
    # Distiller present at serve time, but the store was built without one:
    # simulate by uploading with distiller=None first.
    distiller_holder = {"d": None}

    class SwitchableDistiller:
        subject = None

        def distill(self, key, transcript, date_str, mode="chat"):
            inner = distiller_holder["d"]
            return inner.distill(key, transcript, date_str, mode) if inner else []

    client = _make_client(tmp_path, monkeypatch, distiller=SwitchableDistiller())
    client.post(
        "/api/stores/rescue/upload",
        files=[("files", ("notes.json", CHAT, "application/json"))],
    )
    ov = client.get("/api/stores/rescue/overview").json()
    assert ov["n_atomic"] == 0 and ov["n_verbatim"] == 2
    assert client.get("/api/stores/rescue/partition").json()["lane"] == "verbatim"

    # "Extract atomic facts": now the distiller works
    distiller_holder["d"] = FakeDistiller()
    r = client.post("/api/stores/rescue/distill")
    assert r.status_code == 200
    body = r.json()
    assert body["new_facts"] == 2 and body["n_atomic"] == 2
    # verbatim facts deduped, not duplicated
    assert body["n_facts"] == 4

    # idempotent: a second run adds nothing
    assert client.post("/api/stores/rescue/distill").json()["new_facts"] == 0
    # the map flips to the atomic lane
    assert client.get("/api/stores/rescue/partition").json()["lane"] == "atomic"

    # persisted: reload from disk shows the atomic facts too
    ov = client.get("/api/stores/rescue/overview").json()
    assert ov["n_atomic"] == 2


def test_label_error_returns_http_error(tmp_path, monkeypatch):
    def broken_llm(prompt: str) -> str:
        raise RuntimeError("no API key configured")

    client = _make_client(tmp_path, monkeypatch, distiller=FakeDistiller(), llm_fn=broken_llm)
    client.post(
        "/api/stores/w2/upload",
        files=[("files", ("notes.json", CHAT, "application/json"))],
    )
    r = client.get("/api/stores/w2/partition?label=true")
    assert r.status_code == 502
    assert "no API key configured" in r.json()["detail"]
    # the failure did not poison the label cache
    part = client.get("/api/stores/w2/partition").json()
    assert all(not b["label"] for b in part["buckets"])


def test_legacy_labels_do_not_block_relabeling(tmp_path, monkeypatch):
    calls = []

    def counting_llm(prompt: str) -> str:
        calls.append(prompt)
        return "fresh label"

    client = _make_client(tmp_path, monkeypatch, distiller=FakeDistiller(), llm_fn=counting_llm)
    client.post(
        "/api/stores/w3/upload",
        files=[("files", ("notes.json", CHAT, "application/json"))],
    )
    # simulate a pre-fix cache: labels from the old mixed-lane partition
    from membukkit.storage.localstore import LocalStore

    LocalStore("w3", create=False).update_meta(bucket_labels={"0": "stale mixed-lane label"})

    part = client.get("/api/stores/w3/partition?label=true").json()
    assert calls, "stale lane-less cache must not block relabeling"
    assert part["buckets"][0]["label"] == "fresh label"


CHAT2 = b"""[
  {"date": "2024-05-01", "turns": [
    {"role": "user", "content": "The offsite is in Lisbon this year."},
    {"role": "user", "content": "Marketing booked the venue for June 12th."}
  ]}
]"""


def test_delete_fact_persists_and_keeps_vectors_aligned(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, distiller=FakeDistiller())
    client.post(
        "/api/stores/df/upload",
        files=[("files", ("notes.json", CHAT, "application/json"))],
    )
    facts = client.get("/api/stores/df/facts").json()["facts"]
    assert len(facts) == 4  # 2 verbatim + 2 atomic
    target = facts[0]

    r = client.delete(f"/api/stores/df/facts/{target['id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["deleted"] == target["id"] and body["n_facts"] == 3

    # unknown id -> 404, count unchanged
    assert client.delete("/api/stores/df/facts/nope").status_code == 404
    assert client.get("/api/stores/df/overview").json()["n_facts"] == 3

    # deletion cleared cached bucket labels
    from membukkit.storage.localstore import LocalStore

    meta = LocalStore("df", create=False).meta()
    assert not meta.get("bucket_labels") and not meta.get("bucket_labels_lane")

    # persisted: a fresh app (new hub) reloads 3 facts from disk
    client2 = _make_client(tmp_path, monkeypatch, distiller=FakeDistiller())
    page = client2.get("/api/stores/df/facts").json()
    assert page["total"] == 3
    assert target["id"] not in {f["id"] for f in page["facts"]}

    # vectors stayed aligned with rows: retrieval still works and every
    # evidence id resolves back to a fact with the SAME text.
    ans = client2.post("/api/stores/df/ask", json={"question": "when is the report due?"})
    assert ans.status_code == 200
    evidence = ans.json()["evidence"]
    assert evidence
    for ev in evidence:
        assert ev["fact_id"] != target["id"]
        fact = client2.get(f"/api/stores/df/facts/{ev['fact_id']}/source").json()["fact"]
        assert fact["text"] == ev["text"]


def test_delete_document_removes_its_facts_only(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    client.post(
        "/api/stores/dd/upload",
        files=[("files", ("notes.json", CHAT, "application/json"))],
    )
    client.post(
        "/api/stores/dd/upload",
        files=[("files", ("offsite.json", CHAT2, "application/json"))],
    )
    ov = client.get("/api/stores/dd/overview").json()
    assert ov["n_facts"] == 4 and len(ov["documents"]) == 2
    doc1 = next(d for d in ov["documents"] if d["name"] == "notes.json")

    r = client.delete(f"/api/stores/dd/documents/{doc1['doc_id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["facts_removed"] == 2 and body["n_facts"] == 2

    # registry row and raw source are gone; the other document is intact
    ov = client.get("/api/stores/dd/overview").json()
    assert [d["name"] for d in ov["documents"]] == ["offsite.json"]
    assert client.get(f"/api/stores/dd/documents/{doc1['doc_id']}").status_code == 404
    facts = client.get("/api/stores/dd/facts").json()["facts"]
    assert len(facts) == 2
    assert all(f["doc_name"] == "offsite.json" for f in facts)

    # unknown doc -> 404
    assert client.delete("/api/stores/dd/documents/nope").status_code == 404

    # persisted across a fresh app
    client2 = _make_client(tmp_path, monkeypatch)
    ov2 = client2.get("/api/stores/dd/overview").json()
    assert ov2["n_facts"] == 2 and [d["name"] for d in ov2["documents"]] == ["offsite.json"]


def test_delete_store_returns_404_afterwards(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    client.post(
        "/api/stores/gone/upload",
        files=[("files", ("notes.json", CHAT, "application/json"))],
    )
    assert client.delete("/api/stores/gone").status_code == 200
    assert client.get("/api/stores/gone/overview").status_code == 404
    assert client.delete("/api/stores/gone").status_code == 404
    assert all(s["name"] != "gone" for s in client.get("/api/stores").json()["stores"])


def test_bad_upload_and_missing_store(client):
    assert client.get("/api/stores/nope/overview").status_code == 404
    r = client.post(
        "/api/stores/s1/upload",
        files=[("files", ("weird.xyz", b"data", "application/octet-stream"))],
    )
    assert r.status_code == 200
    assert "error" in r.json()["results"][0]

    # empty store cannot answer
    assert client.post("/api/stores/s1/ask", json={"question": "hi"}).status_code == 400


def test_demos_api_list_and_load(tmp_path, monkeypatch):
    """GET /api/demos lists manifests; POST loads a demo store."""
    from membukkit.cli import demo as demo_mod

    demos = tmp_path / "demos"
    demo_dir = demos / "tiny"
    demo_dir.mkdir(parents=True)
    import json

    (demo_dir / "demo.json").write_text(
        json.dumps(
            {
                "title": "Tiny demo",
                "description": "A one-file demo.",
                "data": ["note.txt"],
                "no_distill": True,
                "questions": ["What about cats?"],
            }
        )
    )
    (demo_dir / "note.txt").write_text("I adopted a tabby cat named Mochi.\n")
    monkeypatch.setattr(demo_mod, "DEMOS_DIR", demos)

    client = _make_client(tmp_path, monkeypatch)
    listed = client.get("/api/demos").json()["demos"]
    assert any(d["id"] == "tiny" for d in listed)
    tiny = next(d for d in listed if d["id"] == "tiny")
    assert tiny["questions"] == ["What about cats?"]

    assert client.post("/api/demos/nope").status_code == 404

    loaded = client.post("/api/demos/tiny").json()
    assert loaded["store"] == "demo-tiny"
    assert loaded["questions"] == ["What about cats?"]
    stores = client.get("/api/stores").json()["stores"]
    assert any(s["name"] == "demo-tiny" for s in stores)


def test_upload_stream_emits_progress_then_result(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, distiller=FakeDistiller())
    with client.stream(
        "POST",
        "/api/stores/work/upload?stream=1",
        files=[("files", ("notes.json", CHAT, "application/json"))],
    ) as res:
        assert res.status_code == 200
        lines = [ln for ln in res.iter_lines() if ln]
    import json

    msgs = [json.loads(ln) for ln in lines]
    assert any(m.get("type") == "progress" for m in msgs)
    result = next(m for m in msgs if m.get("type") == "result")
    assert result["n_facts"] > 0
    assert result["results"][0]["new_facts"] > 0
