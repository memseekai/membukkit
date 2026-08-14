"""PromptConfig wiring: distiller/readers honor overrides; cache keys shift."""

from __future__ import annotations

from membukkit.config import PromptConfig
from membukkit.extraction.distiller import FactDistiller
from membukkit.prompts.extraction import EXTRACTION_PROMPT
from membukkit.prompts.packs import list_prompt_packs, load_prompt_pack
from membukkit.prompts.resolve import (
    PROMPT_VERSION,
    apply_instructions,
    resolve_extraction_template,
    resolve_reader_template,
    validate_prompt_config,
)
from membukkit.prompts.reading import DATED_READER_PROMPT


def test_default_extraction_is_byte_identical_and_stock_version():
    prompts = PromptConfig.default()
    template, ver = resolve_extraction_template(prompts, mode="chat")
    assert template == EXTRACTION_PROMPT
    assert ver == PROMPT_VERSION


def test_extraction_instructions_change_template_and_cache_version():
    prompts = PromptConfig(extraction_instructions="Only extract SKUs.")
    template, ver = resolve_extraction_template(prompts, mode="chat")
    assert "Only extract SKUs." in template
    assert ver.startswith("custom:")
    # Stock path unchanged without instructions.
    stock, stock_ver = resolve_extraction_template(PromptConfig.default(), mode="chat")
    assert stock == EXTRACTION_PROMPT
    assert stock_ver == PROMPT_VERSION


def test_distiller_sends_override_prompt_to_llm():
    seen = []

    def llm(prompt: str) -> str:
        seen.append(prompt)
        return "NONE"

    d = FactDistiller(
        llm,
        prompts=PromptConfig(extraction_instructions="MAGIC_MARKER_SKU_ONLY"),
    )
    d.distill("k1", "[T0] user: hello", "2024-01-01", mode="chat")
    assert seen and "MAGIC_MARKER_SKU_ONLY" in seen[0]


def test_distiller_cache_key_differs_with_custom_prompts():
    def llm(_p: str) -> str:
        return "NONE"

    d0 = FactDistiller(llm, prompts=PromptConfig.default())
    d1 = FactDistiller(
        llm, prompts=PromptConfig(extraction_instructions="custom rules here")
    )
    assert d0._vkey("same") != d1._vkey("same")
    assert d0._vkey("same").startswith(f"{PROMPT_VERSION}:")
    assert d1._vkey("same").startswith("custom:")


def test_reader_instructions_applied_to_stock_dated_template():
    prompts = PromptConfig(reader_instructions="Prefer the newest rent fact.")
    tpl = resolve_reader_template(prompts, "dated")
    assert "Prefer the newest rent fact." in tpl
    assert "{fact_block}" in tpl
    # Full override skips overlay.
    full = PromptConfig(
        dated_reader=DATED_READER_PROMPT,
        reader_instructions="SHOULD_NOT_APPEAR",
    )
    assert "SHOULD_NOT_APPEAR" not in resolve_reader_template(full, "dated")


def test_memory_system_answer_uses_reader_override():
    from datetime import datetime

    from membukkit.config import RetrievalConfig
    from membukkit.pipeline import MemorySystem
    from membukkit.storage.base import Candidate, FactRecord
    from membukkit.storage.memory import InMemoryBackend

    captured = []

    def llm(prompt: str) -> str:
        captured.append(prompt)
        return "ok"

    class _FakeEnc:
        dim = 8

        def encode(self, texts, **kw):
            import numpy as np

            return np.zeros((len(texts), self.dim), dtype="float32")

    class _FakeRerank:
        def score(self, *a, **k):
            return []

    prompts = PromptConfig(reader_instructions="READER_MARKER_XYZ")
    backend = InMemoryBackend(RetrievalConfig(), _FakeEnc())
    backend.upsert_facts(
        [
            FactRecord(
                text="Alex pays $2300 rent",
                timestamp=datetime(2024, 3, 1),
                kind="atomic",
            )
        ]
    )
    mem = MemorySystem(
        encoder=_FakeEnc(),
        reranker=_FakeRerank(),
        llm_fn=llm,
        retrieval=RetrievalConfig(scan_budget=1.0),
        prompts=prompts,
        distiller=None,
        backend=backend,
    )

    def fake_retrieve(*a, **k):
        c = Candidate(
            id="f1",
            text="Alex pays $2300 rent",
            timestamp=datetime(2024, 3, 1),
            kind="atomic",
            cosine=1.0,
        )
        return [], [c], {"n_facts": 1, "n_scanned": 1, "scan_fraction": 1.0, "lanes": {}}

    mem._retrieve_lanes = fake_retrieve  # type: ignore[method-assign]
    mem.answer("How much is rent?", question_date="2024-06-01")
    assert captured and "READER_MARKER_XYZ" in captured[0]


def test_all_shipped_packs_load():
    packs = list_prompt_packs()
    ids = {p["id"] for p in packs}
    assert {
        "personal_assistant",
        "customer_support",
        "contracts",
        "engineering_kb",
        "agent_ops",
    } <= ids
    for p in packs:
        cfg = load_prompt_pack(p["id"])
        assert isinstance(cfg, PromptConfig)
        # Instruction packs should set at least one overlay.
        assert cfg.extraction_instructions or cfg.reader_instructions or cfg.extraction


def test_validate_prompt_config_rejects_missing_placeholders():
    bad = PromptConfig(extraction="no placeholders here")
    try:
        validate_prompt_config(bad)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "transcript" in str(e)


def test_apply_instructions_noop_when_empty():
    assert apply_instructions(EXTRACTION_PROMPT, None) == EXTRACTION_PROMPT
    assert apply_instructions(EXTRACTION_PROMPT, "  ") == EXTRACTION_PROMPT
