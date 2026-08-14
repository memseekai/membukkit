"""CLI add / ask --as-of wiring (no live LLM)."""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pytest

from membukkit.cli import commands
from membukkit.cli.common import open_store, resolve_as_of
from membukkit.config import PromptConfig, RetrievalConfig
from membukkit.pipeline import MemorySystem
from membukkit.storage.base import FactRecord
from membukkit.storage.memory import InMemoryBackend
from membukkit.storage.localstore import LocalStore


class _Enc:
    dim = 8

    def encode(self, texts, normalize=True, show_progress=False):
        if isinstance(texts, str):
            return np.zeros(self.dim, dtype="float32")
        return np.zeros((len(texts), self.dim), dtype="float32")


class _Rerank:
    def score(self, query, texts):
        return np.zeros(len(texts), dtype="float32")


def _mem_with_rent(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMBUKKIT_HOME", str(tmp_path))
    store = LocalStore("s1", create=True)
    backend = InMemoryBackend(RetrievalConfig(scan_budget=1.0), _Enc())
    backend.upsert_facts(
        [
            FactRecord(
                text="rent is 800€",
                timestamp=datetime(2024, 1, 8),
                kind="atomic",
            )
        ]
    )
    mem = MemorySystem(
        encoder=_Enc(),
        reranker=_Rerank(),
        llm_fn=lambda p: "800€",
        retrieval=RetrievalConfig(scan_budget=1.0),
        prompts=PromptConfig.default(),
        distiller=None,
        backend=backend,
    )
    store.save_backend(mem.backend)
    return mem, store


def test_cmd_ask_uses_as_of(tmp_path, monkeypatch, capsys):
    mem, store = _mem_with_rent(tmp_path, monkeypatch)

    def fake_open(name, llm="x", prompt_pack=None, create=False, **kw):
        return mem, store

    monkeypatch.setattr(commands, "open_store", fake_open)
    args = argparse.Namespace(
        store="s1",
        llm="x",
        question="How much is rent?",
        as_of="2024-05-01",
        show_trace=True,
        prompt_pack=None,
    )
    commands.cmd_ask(args)
    out = capsys.readouterr().out
    assert "800" in out
    assert "reader tokens" in out or "trace" in out


def test_resolve_as_of_defaults_to_latest_fact(tmp_path, monkeypatch, capsys):
    mem, _ = _mem_with_rent(tmp_path, monkeypatch)
    assert resolve_as_of(mem, None) == "2024-01-08"
    err = capsys.readouterr().err
    assert "2024-01-08" in err
    assert resolve_as_of(mem, "2024-05-01", announce=False) == "2024-05-01"


def test_open_store_missing_is_friendly(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMBUKKIT_HOME", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        open_store("definitely-missing", create=False)
    msg = str(exc.value)
    assert "not found" in msg
    assert "membukkit stores" in msg
    assert "membukkit add" in msg


def test_cmd_ask_empty_store_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMBUKKIT_HOME", str(tmp_path))
    store = LocalStore("empty", create=True)
    mem = MemorySystem(
        encoder=_Enc(),
        reranker=_Rerank(),
        llm_fn=lambda p: "x",
        retrieval=RetrievalConfig(scan_budget=1.0),
        prompts=PromptConfig.default(),
        distiller=None,
        backend=InMemoryBackend(RetrievalConfig(scan_budget=1.0), _Enc()),
    )

    def fake_open(name, llm="x", prompt_pack=None, create=False, **kw):
        return mem, store

    monkeypatch.setattr(commands, "open_store", fake_open)
    with pytest.raises(SystemExit) as exc:
        commands.cmd_ask(
            argparse.Namespace(
                store="empty",
                llm="x",
                question="hi",
                as_of=None,
                show_trace=False,
                prompt_pack=None,
            )
        )
    assert "membukkit add" in str(exc.value)
