"""ProgressEvent hooks: ingest / distill_store / ProgressFileWriter."""

from __future__ import annotations

import numpy as np
import pytest

from membukkit.config import PromptConfig, RetrievalConfig
from membukkit.pipeline import MemorySystem
from membukkit.progress import ProgressEvent, ProgressFileWriter, emit


class FakeEncoder:
    dim = 8

    def encode(self, texts, normalize=True, show_progress=False):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vecs = []
        for t in items:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(self.dim).astype(np.float32)
            vecs.append(v / (np.linalg.norm(v) + 1e-9))
        out = np.stack(vecs)
        return out[0] if single else out


class FakeReranker:
    def score(self, query, texts):
        return np.zeros(len(texts), dtype=np.float32)


class FakeDistiller:
    subject = None

    def distill(self, key, transcript, date_str, mode="chat"):
        return [(0, f"fact from {date_str}")]


def _mem():
    return MemorySystem(
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        llm_fn=lambda p: "label",
        retrieval=RetrievalConfig(),
        prompts=PromptConfig.default(),
        distiller=FakeDistiller(),
    )


def test_ingest_emits_distill_and_embed_events():
    mem = _mem()
    events: list[ProgressEvent] = []
    n = mem.ingest(
        sessions=[
            [{"role": "user", "content": "Rent is 2100."}],
            [{"role": "user", "content": "Rent is now 2300."}],
        ],
        dates=["2024-01-01", "2024-03-01"],
        doc_name="notes.json",
        on_progress=events.append,
    )
    assert n > 0
    phases = [e.phase for e in events]
    assert "distill" in phases
    assert "embed" in phases
    distill = [e for e in events if e.phase == "distill"]
    assert distill[-1].done == distill[-1].total == 2
    assert any("notes.json" in (e.detail or "") for e in distill)


def test_label_buckets_emits_progress():
    mem = _mem()
    mem.ingest(
        sessions=[[{"role": "user", "content": f"topic fact {i} about apples"}] for i in range(12)],
        dates=["2024-01-01"] * 12,
    )
    events: list[ProgressEvent] = []
    labels = mem.label_buckets(on_progress=events.append)
    assert labels
    assert any(e.phase == "label" for e in events)
    assert events[-1].done == events[-1].total


def test_progress_file_writer_throttles(tmp_path):
    path = tmp_path / "progress.json"
    w = ProgressFileWriter(path, min_interval_s=10.0)
    w.write("distill", 1, 10)
    first = path.read_text()
    w.write("distill", 2, 10)  # throttled
    assert path.read_text() == first
    w.write("distill", 10, 10, force=True)
    assert '"done": 10' in path.read_text()


def test_emit_noop_without_callback():
    emit(None, "distill", 1, 2)  # must not raise


def test_distill_store_forwards_progress(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("MEMBUKKIT_HOME", str(tmp_path))
    from membukkit.cli.common import distill_store
    from membukkit.storage.localstore import LocalStore

    store = LocalStore("p", create=True)
    store.add_document(
        "a.txt",
        [[{"role": "user", "content": "Hello world fact one."}]],
        ["2024-01-01"],
        doc_type="document",
    )
    mem = _mem()
    # Seed verbatim so store has content; distill_store re-ingests docs.
    events: list[ProgressEvent] = []
    n = distill_store(mem, store, on_progress=events.append)
    assert n >= 0
    assert any(e.phase in ("distill", "embed") for e in events)
