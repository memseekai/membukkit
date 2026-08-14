"""Service-layer tests: route wiring via a stub MemoryService (no models/DB)."""

from __future__ import annotations

from datetime import datetime

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from membukkit.pipeline import AnswerResult, RetrievalTrace  # noqa: E402
from membukkit.service.manager import namespace_for  # noqa: E402


class FakeBackend:
    def __init__(self):
        self.facts = 0

    def count(self):
        return self.facts


class FakeMem:
    def __init__(self):
        self._backend = FakeBackend()
        self.ingested = []

    def ingest(self, sessions, dates=None, subject=None):
        if dates is not None:
            assert all(d is None or isinstance(d, datetime) for d in dates)
        self.ingested.append((sessions, dates, subject))
        self._backend.facts += sum(len(s) for s in sessions)

    def answer(self, question, question_date="", identity="", generate_answer=True):
        tr = RetrievalTrace(
            opened_buckets=[{"bucket": 1, "route_prob": 0.6, "size": 4}],
            scan_fraction=0.33,
            n_facts=self._backend.facts,
            n_scanned=4,
            reader_type="dated",
            ranked_fact_times=["2024-06-01T00:00:00"],
            backend="turbopuffer",
            perf={"cache_temperature": "warm"},
        )
        return AnswerResult(
            answer=f"answer to: {question}" if generate_answer else None,
            trace=tr,
            facts=["[2024-06-01] a fact"],
        )

    def partition(self):
        return {"k_eff": 3, "version": 1, "sizes": {0: 2, 1: 4, 2: 1}}

    def label_buckets(self):
        return {0: "diet", 1: "work", 2: "travel"}


class _StubConfig:
    telemetry = False  # keep create_app from configuring global telemetry in tests
    environment = None
    capture_content = False


class StubService:
    def __init__(self):
        self.config = _StubConfig()
        self.mems = {}
        self.warmed = []
        self.deleted = []

    def get(self, owner):
        return self.mems.setdefault(owner, FakeMem())

    def warm(self, owner):
        self.warmed.append(owner)

    def recluster(self, owner):
        return True

    def delete(self, owner):
        self.deleted.append(owner)


@pytest.fixture
def client():
    from membukkit.service.app import create_app

    return TestClient(create_app(service=StubService()))  # ty: ignore[invalid-argument-type]  # structural test double


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ingest_then_answer_flow(client):
    r = client.post(
        "/v1/alice/ingest",
        json={
            "sessions": [[{"role": "user", "content": "I switched to a vegan diet."}]],
            "dates": ["2024-06-01T10:30:00Z"],
        },
    )
    assert r.status_code == 200
    assert r.json()["n_facts"] == 1

    r = client.post(
        "/v1/alice/answer",
        json={
            "question": "What diet?",
            "question_date": "2024-07-01T09:00:00-05:00",
            "trace": True,
        },
    )
    body = r.json()
    assert body["answer"].startswith("answer to:")
    assert body["facts"] == ["[2024-06-01] a fact"]
    assert body["trace"]["scan_fraction"] == 0.33
    assert body["trace"]["backend"] == "turbopuffer"
    assert body["trace"]["ranked_fact_times"] == ["2024-06-01T00:00:00"]
    assert body["trace"]["perf"]["cache_temperature"] == "warm"
    assert body["trace"]["opened_buckets"][0]["bucket"] == 1


def test_answer_defaults_omit_trace(client):
    """Default request returns answer + facts, no trace key."""
    r = client.post("/v1/alice/answer", json={"question": "What diet?"})
    body = r.json()
    assert body["answer"].startswith("answer to:")
    assert body["facts"] == ["[2024-06-01] a fact"]
    assert "trace" not in body


def test_answer_facts_only_skips_answer(client):
    """answer=false yields a clean facts-only body (no answer, no trace)."""
    r = client.post("/v1/alice/answer", json={"question": "What diet?", "answer": False})
    body = r.json()
    assert body == {"facts": ["[2024-06-01] a fact"]}


def test_legacy_dates_still_parse(client):
    r = client.post(
        "/v1/legacy/ingest",
        json={
            "sessions": [[{"role": "user", "content": "A legacy dated fact."}]],
            "dates": ["2024/06/01"],
        },
    )
    assert r.status_code == 200
    r = client.post(
        "/v1/legacy/answer",
        json={
            "question": "What fact?",
            "question_date": "2024/07/01",
        },
    )
    assert r.status_code == 200


def test_partition_and_labels(client):
    assert client.get("/v1/bob/partition").json()["k_eff"] == 3
    assert client.post("/v1/bob/label_buckets").json()["1"] == "work"


def test_warm_recluster_delete(client):
    assert client.post("/v1/carol/warm").json()["ok"] is True
    assert client.post("/v1/carol/recluster").json()["reclustered"] is True
    assert client.delete("/v1/carol").json()["ok"] is True


def test_namespace_sanitization():
    assert namespace_for("user-42!!") == "mem_user-42"
    assert namespace_for("  Bob Smith  ") == "mem_Bob_Smith"
    with pytest.raises(ValueError):
        namespace_for("")
