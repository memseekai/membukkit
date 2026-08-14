"""Tests for bundled demo loading and UI deep-link helpers."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from membukkit.cli import demo as demo_mod
from membukkit.cli.ui import register as register_ui
from membukkit.cli.ui import ui_url
from membukkit.config import PromptConfig, RetrievalConfig
from membukkit.pipeline import MemorySystem


class FakeEncoder:
    dim = 16

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


def _fake_build_system(llm="x", encoder_spec="y", distill=True, retrieval=None, prompts=None):
    return MemorySystem(
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        llm_fn=lambda p: "ans",
        retrieval=retrieval or RetrievalConfig(),
        prompts=prompts or PromptConfig.default(),
        distiller=None,
    )


@pytest.fixture()
def tiny_demos(tmp_path, monkeypatch):
    demos = tmp_path / "demos"
    demo_dir = demos / "tiny"
    demo_dir.mkdir(parents=True)
    (demo_dir / "demo.json").write_text(
        json.dumps(
            {
                "title": "Tiny demo",
                "description": "A one-file verbatim demo.",
                "data": ["note.txt"],
                "no_distill": True,
                "questions": ["What about cats?"],
            }
        )
    )
    (demo_dir / "note.txt").write_text(
        "I adopted a tabby cat named Mochi in March 2024.\n"
    )
    monkeypatch.setattr(demo_mod, "DEMOS_DIR", demos)
    monkeypatch.setenv("MEMBUKKIT_HOME", str(tmp_path / "home"))

    import membukkit.cli.common as common

    monkeypatch.setattr(common, "build_system", _fake_build_system)
    return demos


def test_ensure_demo_store_unknown(tiny_demos):
    with pytest.raises(ValueError, match="unknown demo"):
        demo_mod.ensure_demo_store("nope", llm="openai:gpt-4o-mini")


def test_ensure_demo_store_ingests_and_reuses(tiny_demos, capsys):
    store = demo_mod.ensure_demo_store("tiny", llm="openai:gpt-4o-mini")
    assert store == "demo-tiny"
    n = demo_mod._store_fact_count(store)
    assert n is not None and n > 0

    out1 = capsys.readouterr().out
    assert "Tiny demo" in out1 or "document" in out1

    store2 = demo_mod.ensure_demo_store("tiny", llm="openai:gpt-4o-mini")
    assert store2 == "demo-tiny"
    out2 = capsys.readouterr().out
    assert "using existing store demo-tiny" in out2
    assert f"({n} facts)" in out2


def test_available_demos_lists_tiny(tiny_demos):
    demos = demo_mod.available_demos()
    assert "tiny" in demos
    assert demos["tiny"]["title"] == "Tiny demo"


def test_ui_argparse_accepts_demo():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    register_ui(sub)
    args = parser.parse_args(["ui", "--demo", "personal-assistant", "--no-browser"])
    assert args.demo == "personal-assistant"
    assert args.no_browser is True


def test_demo_argparse_accepts_ui():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    demo_mod.register(sub)
    args = parser.parse_args(["demo", "personal-assistant", "--ui", "--no-browser"])
    assert args.name == "personal-assistant"
    assert args.ui is True
    assert args.no_browser is True


def test_ui_url_with_store_and_tab():
    assert ui_url("127.0.0.1", 8377) == "http://127.0.0.1:8377"
    assert (
        ui_url("127.0.0.1", 8377, store="demo-personal-assistant", tab="ask")
        == "http://127.0.0.1:8377/?store=demo-personal-assistant&tab=ask"
    )


def test_packaged_demos_discoverable():
    """Wheel/editable installs expose the shipped demos."""
    assert demo_mod.DEMOS_DIR.is_dir()
    demos = demo_mod.available_demos()
    assert set(demos) >= {
        "personal-assistant",
        "support-brain",
        "contract-qa",
        "engineering-kb",
        "agent-ops",
    }
    assert demos["personal-assistant"].get("question_date")
    assert demos["contract-qa"].get("ask_callouts")
    assert demos["agent-ops"].get("prompt_pack") == "agent_ops"
