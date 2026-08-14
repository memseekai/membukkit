"""Memory facade, WriteReport, supersession, and local v1 API."""

from __future__ import annotations

import numpy as np
import pytest

from membukkit import Memory, WriteReport
from membukkit.config import PromptConfig, RetrievalConfig
from membukkit.pipeline import MemorySystem
from membukkit.supersession import fact_status, is_active_as_of, link_supersessions


class FakeEncoder:
    dim = 16

    def encode(self, texts, normalize=True, show_progress=False):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        rent_base = None
        vecs = []
        for t in items:
            # Rent facts share a base vector so supersession can link them.
            if "rent" in t.lower():
                if rent_base is None:
                    rng = np.random.default_rng(1)
                    rent_base = rng.standard_normal(self.dim).astype(np.float32)
                    rent_base /= np.linalg.norm(rent_base) + 1e-9
                jitter = np.random.default_rng(abs(hash(t)) % (2**32)).standard_normal(
                    self.dim
                ).astype(np.float32)
                v = rent_base * 0.92 + jitter * 0.08
            else:
                rng = np.random.default_rng(abs(hash(t)) % (2**32))
                v = rng.standard_normal(self.dim).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-9)
            vecs.append(v)
        out = np.stack(vecs)
        return out[0] if single else out


class FakeReranker:
    def score(self, query, texts):
        return np.zeros(len(texts), dtype=np.float32)


class FakeDistiller:
    subject = None

    def distill(self, key, transcript, date_str, mode="chat"):
        # Emit one atomic fact from the transcript body.
        body = transcript.replace("[T0]", "").strip()
        if "NONE" in body.upper() and "rent" not in body.lower():
            return []
        # Prefer the last user-looking line.
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        text = lines[-1] if lines else body
        text = text if "noted" in text else f"{text} (noted {date_str})"
        return [(0, text)]


def _mem(distiller=None) -> MemorySystem:
    return MemorySystem(
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        llm_fn=lambda p: "Rent is $2300 after the March raise.",
        retrieval=RetrievalConfig(),
        prompts=PromptConfig.default(),
        distiller=distiller if distiller is not None else FakeDistiller(),
    )


def test_write_report_int_compatible():
    r = WriteReport(n_stored=3, status="ok")
    assert int(r) == 3
    assert r
    assert r + 2 == 5
    assert sum([r, WriteReport(n_stored=1)], 0) == 4


def test_memory_add_ask_rent_story():
    mem = Memory.wrap(_mem())
    r1 = mem.add("rent is $2100", subject="Alex", date="2024-01-10")
    r2 = mem.add("rent raised to $2300", subject="Alex", date="2024-03-01")
    assert r1.status == "ok" and r2.status == "ok"
    assert int(r1) + int(r2) > 0
    rows = mem.backend.list_atomic_rows()
    assert len(rows) >= 1
    if len(rows) >= 2:
        pairs = link_supersessions(mem.backend, [rows[-1]["id"]], threshold=0.5)
        assert isinstance(pairs, list)

    receipt = mem.ask("How much is my rent?", as_of="2024-06-01")
    assert receipt.answer
    assert receipt.scan_fraction >= 0
    assert receipt.evidence


def test_empty_extract_status():
    class EmptyDistiller:
        subject = None

        def distill(self, key, transcript, date_str, mode="chat"):
            return []

    mem = Memory.wrap(_mem(distiller=EmptyDistiller()))
    # Union mode still stores verbatim turns, so status may be ok with verbatim.
    # Disable union to force empty_extract when distiller returns nothing.
    mem.system._retrieval.union = False
    report = mem.add("hello there", subject="u", date="2024-01-01")
    assert report.status == "empty_extract"
    assert report.n_stored == 0
    assert report.warnings


def test_is_active_as_of_supersession():
    from datetime import datetime

    old_ts = datetime(2024, 1, 10)
    new_ts = datetime(2024, 3, 1)
    assert is_active_as_of(
        superseded_by="new",
        valid_to=new_ts,
        timestamp=old_ts,
        as_of=datetime(2024, 2, 1),
    )
    assert not is_active_as_of(
        superseded_by="new",
        valid_to=new_ts,
        timestamp=old_ts,
        as_of=datetime(2024, 6, 1),
    )
    assert (
        fact_status(
            superseded_by="new",
            valid_to=new_ts,
            timestamp=old_ts,
            as_of=datetime(2024, 6, 1),
        )
        == "superseded"
    )


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMBUKKIT_HOME", str(tmp_path))

    def fake_build_system(
        llm="x", encoder_spec="y", distill=True, retrieval=None, prompts=None
    ):
        return MemorySystem(
            encoder=FakeEncoder(),
            reranker=FakeReranker(),
            llm_fn=lambda p: "synthetic",
            retrieval=retrieval or RetrievalConfig(),
            prompts=prompts or PromptConfig.default(),
            distiller=FakeDistiller(),
        )

    import membukkit.cli.common as common

    monkeypatch.setattr(common, "build_system", fake_build_system)
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from membukkit.service.local_app import create_local_app

    return TestClient(create_local_app())


def test_v1_add_search_ask(client):
    r = client.post(
        "/api/v1/agent/add",
        json={"content": "rent is $2100", "subject": "alex", "date": "2024-01-10"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_stored"] > 0

    client.post(
        "/api/v1/agent/add",
        json={"content": "rent is now $2300", "subject": "alex", "date": "2024-03-01"},
    )

    s = client.post(
        "/api/v1/agent/search",
        json={"query": "how much rent", "as_of": "2024-06-01", "include_history": True},
    )
    assert s.status_code == 200
    assert s.json()["hits"]

    a = client.post(
        "/api/v1/agent/ask",
        json={"query": "how much is rent?", "as_of": "2024-06-01"},
    )
    assert a.status_code == 200
    assert a.json()["answer"]
    assert "scan_fraction" in a.json()["trace"]
