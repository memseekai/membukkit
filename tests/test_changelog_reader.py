"""Changelog query detection and reader overlay (memory-timeline recency)."""

from membukkit.config import PromptConfig
from membukkit.prompts.reading import (
    CHANGELOG_READER_OVERLAY,
    DATED_READER_PROMPT,
    REASONING_READER_PROMPT,
)
from membukkit.prompts.resolve import resolve_reader_template
from membukkit.retrieval.router import is_changelog_query


def test_is_changelog_query_matches_common_phrasings():
    assert is_changelog_query("What changed recently?")
    assert is_changelog_query("What has changed over time?")
    assert is_changelog_query("What's new?")
    assert is_changelog_query("what was updated recently")
    assert not is_changelog_query("How much is my rent?")
    assert not is_changelog_query("Which days do I go to the gym?")


def test_changelog_overlay_injected_for_reasoning_and_dated():
    prompts = PromptConfig.default()
    reason = resolve_reader_template(prompts, "reasoning", changelog=True)
    dated = resolve_reader_template(prompts, "dated", changelog=True)
    assert "memory timeline" in reason
    assert "Do NOT reply N/I merely" in reason or "do not reply N/I merely" in reason.lower()
    assert CHANGELOG_READER_OVERLAY.split()[0] in reason  # "Changelog"
    assert "memory timeline" in dated
    # Stock templates unchanged without the flag.
    assert resolve_reader_template(prompts, "reasoning") == REASONING_READER_PROMPT
    assert resolve_reader_template(prompts, "dated") == DATED_READER_PROMPT


def test_effective_date_framing_still_present_for_current_state():
    prompts = PromptConfig.default()
    reason = resolve_reader_template(prompts, "reasoning", changelog=False)
    dated = resolve_reader_template(prompts, "dated", changelog=False)
    assert "in effect as of today" in reason
    assert "takes effect AFTER today" in dated
    assert "memory timeline" not in reason
    assert "memory timeline" not in dated


def test_changelog_plus_pack_instructions_stack():
    prompts = PromptConfig(reader_instructions="PACK_MARKER_RENT")
    tpl = resolve_reader_template(prompts, "reasoning", changelog=True)
    assert "PACK_MARKER_RENT" in tpl
    assert "memory timeline" in tpl


def test_pipeline_injects_changelog_overlay_on_ask(monkeypatch):
    from datetime import datetime

    from membukkit.config import RetrievalConfig
    from membukkit.pipeline import MemorySystem
    from membukkit.storage.base import Candidate, FactRecord
    from membukkit.storage.memory import InMemoryBackend

    captured = []

    def llm(prompt: str) -> str:
        captured.append(prompt)
        return "Answer: rent rose; gym days shifted"

    class _Enc:
        dim = 8

        def encode(self, texts, normalize=True, show_progress=False):
            import numpy as np

            n = 1 if isinstance(texts, str) else len(texts)
            return np.zeros((n, self.dim), dtype="float32")

    class _Rerank:
        def score(self, *a, **k):
            return []

    backend = InMemoryBackend(RetrievalConfig(), _Enc())
    backend.upsert_facts(
        [
            FactRecord(
                text="rent is 800€",
                timestamp=datetime(2024, 1, 8),
                kind="atomic",
            ),
            FactRecord(
                text="rent is going up to 950€ from June",
                timestamp=datetime(2024, 4, 2),
                kind="atomic",
            ),
        ]
    )
    mem = MemorySystem(
        encoder=_Enc(),
        reranker=_Rerank(),
        llm_fn=llm,
        retrieval=RetrievalConfig(scan_budget=1.0),
        prompts=PromptConfig.default(),
        distiller=None,
        backend=backend,
    )

    def fake_retrieve(*a, **k):
        cands = [
            Candidate(
                id="a",
                text="rent is 800€",
                timestamp=datetime(2024, 1, 8),
                kind="atomic",
                cosine=1.0,
            ),
            Candidate(
                id="b",
                text="rent is going up to 950€ from June",
                timestamp=datetime(2024, 4, 2),
                kind="atomic",
                cosine=0.9,
            ),
        ]
        return [], cands, {
            "n_facts": 2,
            "n_scanned": 2,
            "scan_fraction": 1.0,
            "lanes": {},
        }

    mem._retrieve_lanes = fake_retrieve  # type: ignore[method-assign]
    out = mem.answer("What changed recently?", question_date="2026-08-07")
    assert "rent" in out.answer.lower() or "950" in out.answer
    assert captured and "memory timeline" in captured[0]
