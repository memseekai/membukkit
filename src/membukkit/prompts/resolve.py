"""Resolve PromptConfig into concrete extraction / reader templates."""

from __future__ import annotations

import hashlib
from typing import Optional, Tuple

from membukkit.config import PromptConfig
from membukkit.prompts.extraction import (
    DOCUMENT_EXTRACTION_PROMPT,
    EXTRACTION_PROMPT,
    NAMED_EXTRACTION_PROMPT,
)
from membukkit.prompts.reading import (
    ABSTAIN_GATE_PROMPT,
    CHANGELOG_READER_OVERLAY,
    DATED_READER_PROMPT,
    REASONING_READER_PROMPT,
    RECOMMENDATION_READER_PROMPT,
)

# Built-in version tags kept for distill-cache compatibility when the user has
# not customized anything (benchmarks and existing caches stay valid).
PROMPT_VERSION = "v3"
NAMED_PROMPT_VERSION = "v4n"
DOC_PROMPT_VERSION = "d1"

PLACEHOLDERS = {
    "extraction": ("{date}", "{transcript}"),
    "extraction_named": ("{subject}", "{date}", "{transcript}"),
    "extraction_document": ("{date}", "{transcript}"),
    "dated_reader": ("{identity_preamble}", "{today_line}", "{fact_block}", "{question}"),
    "recommendation_reader": (
        "{identity_preamble}",
        "{today_line}",
        "{fact_block}",
        "{question}",
    ),
    "reasoning_reader": ("{identity_preamble}", "{today_line}", "{fact_block}", "{question}"),
}


def apply_instructions(template: str, instructions: Optional[str]) -> str:
    """Append natural-language instructions into a prompt template.

    Inserts before the conversation / facts block when present so the model
    sees the overlay with the rules. Empty instructions leave the template
    byte-identical (required for default distill-cache keys).
    """
    text = (instructions or "").strip()
    if not text:
        return template
    block = f"\nAdditional instructions:\n{text}\n"
    for marker in (
        "\nConversation (occurred on",
        "\nConversation:",
        "\nFacts (one",
        "\nMemories:",
        "\nQuestion:",
    ):
        if marker in template:
            return template.replace(marker, block + marker, 1)
    return template + block


def _fingerprint(template: str) -> str:
    return hashlib.sha1(template.encode("utf-8")).hexdigest()[:8]


def resolve_extraction_template(
    prompts: PromptConfig,
    mode: str = "chat",
    subject: Optional[str] = None,
) -> Tuple[str, str]:
    """Return ``(template, cache_version_prefix)`` for distillation.

    ``cache_version_prefix`` is the built-in tag when using stock prompts with
    no overlays; otherwise ``custom:<hash>`` so custom prompts never read the
    default distill cache.
    """
    if mode == "document":
        base = prompts.extraction_document or DOCUMENT_EXTRACTION_PROMPT
        stock = DOCUMENT_EXTRACTION_PROMPT
        stock_ver = DOC_PROMPT_VERSION
    elif subject:
        base = prompts.extraction_named or NAMED_EXTRACTION_PROMPT
        stock = NAMED_EXTRACTION_PROMPT
        stock_ver = NAMED_PROMPT_VERSION
    else:
        base = prompts.extraction or EXTRACTION_PROMPT
        stock = EXTRACTION_PROMPT
        stock_ver = PROMPT_VERSION

    template = apply_instructions(base, prompts.extraction_instructions)
    if template == stock and not (prompts.extraction_instructions or "").strip():
        return template, stock_ver
    return template, f"custom:{_fingerprint(template)}"


def resolve_reader_template(
    prompts: PromptConfig,
    kind: str,
    *,
    changelog: bool = False,
) -> str:
    """Resolve a reader template by kind: dated | reasoning | recommendation | abstain.

    When ``changelog`` is True (dated/reasoning only), append the changelog
    overlay so "recently" means the memory timeline, not wall-clock proximity.
    """
    mapping = {
        "dated": (prompts.dated_reader, DATED_READER_PROMPT),
        "reasoning": (prompts.reasoning_reader, REASONING_READER_PROMPT),
        "recommendation": (prompts.recommendation_reader, RECOMMENDATION_READER_PROMPT),
        "abstain": (prompts.abstain_gate, ABSTAIN_GATE_PROMPT),
    }
    if kind not in mapping:
        raise ValueError(f"unknown reader kind {kind!r}")
    override, stock = mapping[kind]
    base = override or stock
    # Full overrides skip the shared reader_instructions overlay so packs that
    # ship complete templates are not double-annotated. Instruction-only packs
    # use stock + overlay.
    if not override:
        base = apply_instructions(base, prompts.reader_instructions)
    if changelog and kind in ("dated", "reasoning"):
        base = apply_instructions(base, CHANGELOG_READER_OVERLAY)
    return base


def validate_prompt_config(prompts: PromptConfig) -> None:
    """Raise ValueError if an override is missing required placeholders."""
    checks = [
        ("extraction", prompts.extraction, PLACEHOLDERS["extraction"]),
        ("extraction_named", prompts.extraction_named, PLACEHOLDERS["extraction_named"]),
        ("extraction_document", prompts.extraction_document, PLACEHOLDERS["extraction_document"]),
        ("dated_reader", prompts.dated_reader, PLACEHOLDERS["dated_reader"]),
        ("recommendation_reader", prompts.recommendation_reader, PLACEHOLDERS["recommendation_reader"]),
        ("reasoning_reader", prompts.reasoning_reader, PLACEHOLDERS["reasoning_reader"]),
    ]
    for name, value, required in checks:
        if not value:
            continue
        missing = [p for p in required if p not in value]
        if missing:
            raise ValueError(f"{name} template missing placeholders: {', '.join(missing)}")
