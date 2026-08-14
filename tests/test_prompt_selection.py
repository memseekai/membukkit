"""Extraction prompt selection: assistant chats keep the user-centric prompt,
file-ingested documents/multi-speaker chats get the subject-agnostic one.

The bug this pins: a WhatsApp group export ingested as role-"user" turns was
distilled with the "facts about the USER" chat prompt, which answers NONE for
multi-speaker content — zero atomic facts, cached forever.
"""

from __future__ import annotations

import numpy as np

from membukkit.config import PromptConfig, RetrievalConfig
from membukkit.extraction.distiller import (
    DOC_PROMPT_VERSION,
    FactDistiller,
    PROMPT_VERSION,
)
from membukkit.pipeline import MemorySystem
from membukkit.storage.memory import InMemoryBackend


class RecordingLLM:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "0 | a fact"


class FakeEncoder:
    dim = 8

    def encode(self, texts, normalize=True, show_progress=False):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        rng = [abs(hash(t)) % 997 for t in items]
        vecs = np.stack([np.linspace(r, r + 1, self.dim) for r in rng]).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs[0] if single else vecs


class FakeReranker:
    def score(self, query, texts):
        return np.zeros(len(texts), dtype=np.float32)


def _mem(llm):
    cfg = RetrievalConfig(union=True)
    encoder = FakeEncoder()
    return MemorySystem(
        encoder=encoder,
        reranker=FakeReranker(),
        llm_fn=llm,
        retrieval=cfg,
        prompts=PromptConfig.default(),
        distiller=FactDistiller(llm),
        backend=InMemoryBackend(cfg, encoder),
    )


ASSISTANT_SESSION = [
    {"role": "user", "content": "I adopted a dog named Biscuit"},
    {"role": "assistant", "content": "Congrats! Golden retrievers are great."},
]
SPEAKER_SESSION = [
    {"role": "user", "content": "Gaia: Beach day on Saturday?"},
    {"role": "user", "content": "Zak: I'm in, I'll bring the frisbee"},
]


def test_assistant_session_uses_user_prompt():
    llm = RecordingLLM()
    _mem(llm).ingest([ASSISTANT_SESSION], doc_type="chat")
    assert len(llm.prompts) == 1
    assert "about the USER" in llm.prompts[0]


def test_speaker_only_session_uses_document_prompt():
    llm = RecordingLLM()
    _mem(llm).ingest([SPEAKER_SESSION], dates=["2026-04-18"], doc_type="chat")
    assert len(llm.prompts) == 1
    assert "ATTRIBUTE EVERY STATEMENT TO ITS NAMED SOURCE" in llm.prompts[0]
    assert "about the USER" not in llm.prompts[0]


def test_document_doc_type_uses_document_prompt():
    llm = RecordingLLM()
    _mem(llm).ingest(
        [[{"role": "user", "content": "Q3 revenue grew 12% to $4.1M."}]],
        doc_type="document",
    )
    assert "ATTRIBUTE EVERY STATEMENT TO ITS NAMED SOURCE" in llm.prompts[0]


def test_programmatic_ingest_keeps_chat_prompt_and_cache_keys():
    # No doc_type (eval/bench/API path): even speaker-only sessions keep the
    # existing chat prompt, so benchmark behaviour and caches are untouched.
    llm = RecordingLLM()
    _mem(llm).ingest([SPEAKER_SESSION])
    assert "about the USER" in llm.prompts[0]

    d = FactDistiller(lambda p: "NONE")
    assert d._vkey("k") == f"{PROMPT_VERSION}:k"
    assert d._vkey("k", "document") == f"{DOC_PROMPT_VERSION}:k"
    assert d._vkey("k", "document") != d._vkey("k")


def test_distiller_modes_are_cached_independently():
    llm = RecordingLLM()
    d = FactDistiller(llm)
    d.distill("k", "[T0] user: Gaia: hello", "2026-04-18")
    d.distill("k", "[T0] user: Gaia: hello", "2026-04-18", mode="document")
    assert len(llm.prompts) == 2, "document mode must not hit the chat-mode cache"
    # and each mode is itself cached
    d.distill("k", "[T0] user: Gaia: hello", "2026-04-18", mode="document")
    assert len(llm.prompts) == 2
