"""Prompt editor API: GET/PUT/pack/reset + hub cache drop on change."""

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


@pytest.fixture()
def client_and_seen(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMBUKKIT_HOME", str(tmp_path))
    seen: dict = {}

    def fake_build_system(
        llm="x",
        encoder_spec="y",
        distill=True,
        retrieval=None,
        prompts=None,
    ):
        prompts = prompts or PromptConfig.default()
        seen["prompts"] = prompts
        return MemorySystem(
            encoder=FakeEncoder(),
            reranker=FakeReranker(),
            llm_fn=lambda _p: "ok",
            retrieval=retrieval or RetrievalConfig(),
            prompts=prompts,
            distiller=None,
        )

    import membukkit.cli.common as common

    monkeypatch.setattr(common, "build_system", fake_build_system)

    from membukkit.service.local_app import create_local_app

    return TestClient(create_local_app()), seen


@pytest.fixture()
def client(client_and_seen):
    return client_and_seen[0]


def _ensure_store(client: TestClient, name: str = "work") -> None:
    r = client.post(f"/api/stores/{name}")
    assert r.status_code == 200


def test_get_returns_packs_and_is_default(client):
    _ensure_store(client)
    r = client.get("/api/stores/work/prompts")
    assert r.status_code == 200
    body = r.json()
    assert body["is_default"] is True
    assert body["pack_id"] is None
    assert body["prompts"] == {}
    pack_ids = {p["id"] for p in body["packs"]}
    assert "customer_support" in pack_ids
    assert "extraction" in body["placeholders"]
    assert "{transcript}" in body["placeholders"]["extraction"]


def test_put_persists_and_hub_reopen_sees_prompts(client_and_seen):
    client, seen = client_and_seen
    _ensure_store(client)

    r = client.put(
        "/api/stores/work/prompts",
        json={"extraction_instructions": "Prefer ticket ids and error codes."},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["is_default"] is False
    assert body["pack_id"] is None
    assert body["prompts"]["extraction_instructions"] == "Prefer ticket ids and error codes."

    # Drop already happened in PUT; next open must rebuild with meta prompts.
    seen.clear()
    ov = client.get("/api/stores/work/overview")
    assert ov.status_code == 200
    assert seen["prompts"].extraction_instructions == "Prefer ticket ids and error codes."


def test_post_pack_customer_support(client):
    _ensure_store(client)
    r = client.post("/api/stores/work/prompts/pack", json={"pack_id": "customer_support"})
    assert r.status_code == 200
    body = r.json()
    assert body["pack_id"] == "customer_support"
    assert body["is_default"] is False
    assert "extraction_instructions" in body["prompts"]


def test_post_unknown_pack_404(client):
    _ensure_store(client)
    r = client.post("/api/stores/work/prompts/pack", json={"pack_id": "no_such_pack"})
    assert r.status_code == 404


def test_post_reset_clears(client):
    _ensure_store(client)
    client.put(
        "/api/stores/work/prompts",
        json={"reader_instructions": "Be terse."},
    )
    client.post("/api/stores/work/prompts/pack", json={"pack_id": "customer_support"})
    r = client.post("/api/stores/work/prompts/reset")
    assert r.status_code == 200
    body = r.json()
    assert body["is_default"] is True
    assert body["pack_id"] is None
    assert body["prompts"] == {}


def test_put_broken_template_400(client):
    _ensure_store(client)
    r = client.put(
        "/api/stores/work/prompts",
        json={"extraction": "Extract facts from this text with no placeholders."},
    )
    assert r.status_code == 400
    assert "transcript" in r.json()["detail"].lower() or "placeholder" in r.json()["detail"].lower()
