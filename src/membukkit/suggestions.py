"""Generate grounded Ask suggestion chips from store facts."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_SAMPLE = 40
_MAX_Q = 5
_MIN_Q = 3
_MAX_Q_LEN = 160

_SUGGEST_PROMPT = """You write short follow-up questions for a memory store.

Given ONLY the dated facts below, return a JSON array of {n} questions a user
could ask. Rules:
- Every question MUST be answerable from the listed facts (names, amounts, dates).
- Prefer concrete lookups and supersession ("what replaced X?", "how much is Y?").
- Do NOT ask vague summary questions (no "most important", "summarize", "preferences").
- One short sentence each. No numbering outside the JSON.

Facts:
{facts}

Return ONLY a JSON array of strings, e.g. ["Question one?", "Question two?"].
"""


def _sample_facts(backend: Any, limit: int = _SAMPLE) -> List[Dict]:
    if getattr(backend, "count", lambda: 0)() == 0:
        return []
    facts_page = getattr(backend, "facts_page", None)
    if not callable(facts_page):
        return []

    def _pull(kind: Optional[str]) -> List[Dict]:
        page = facts_page(offset=0, limit=200, kind=kind)
        rows = list(page.get("facts") or [])
        return [f for f in rows if f.get("status") != "superseded"]

    facts = _pull("atomic")
    if len(facts) < 8:
        facts = _pull(None)
    facts.sort(key=lambda f: f.get("timestamp") or "", reverse=True)
    return facts[:limit]


def _format_facts(facts: Sequence[Dict]) -> str:
    lines = []
    for f in facts:
        ts = (f.get("timestamp") or "")[:10] or "?"
        text = (f.get("text") or "").strip().replace("\n", " ")
        if len(text) > 180:
            text = text[:177] + "…"
        if not text:
            continue
        lines.append(f"- [{ts}] {text}")
    return "\n".join(lines)


def _extract_json_array(raw: str) -> Optional[list]:
    text = (raw or "").strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, list) else None
        except json.JSONDecodeError:
            return None
    return None


def _normalize_questions(items: list, *, n: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if not isinstance(item, str):
            continue
        q = " ".join(item.strip().split())
        if not q or len(q) > _MAX_Q_LEN:
            continue
        if not q.endswith("?"):
            q = q.rstrip(".") + "?"
        key = q.lower()
        if key in seen:
            continue
        # Ban vague summary prompts.
        bad = ("most important", "summarize", "summary of", "what preferences")
        if any(b in key for b in bad):
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= n:
            break
    return out


def suggest_questions(
    backend: Any,
    llm_fn: Callable[[str], str],
    *,
    n: int = _MAX_Q,
) -> List[str]:
    """Return 3–5 grounded Ask chips from a sample of current facts.

    On empty store, LLM failure, or unparseable output, returns ``[]``.
    """
    n = max(_MIN_Q, min(int(n), _MAX_Q))
    facts = _sample_facts(backend)
    if not facts:
        return []
    block = _format_facts(facts)
    if not block.strip():
        return []
    prompt = _SUGGEST_PROMPT.format(n=n, facts=block)
    try:
        from membukkit.usage import get_meter

        get_meter().take()
        raw = llm_fn(prompt)
    except Exception:
        logger.exception("suggest_questions LLM call failed")
        return []
    data = _extract_json_array(raw if isinstance(raw, str) else str(raw))
    if not data:
        return []
    return _normalize_questions(data, n=n)


def refresh_store_suggestions(
    store: Any,
    mem: Any,
    llm_spec: str = "",
) -> List[str]:
    """Generate chips, persist on store meta when non-empty, record usage.

    Never raises; leaves prior ``suggested_questions`` if generation fails.
    """
    from membukkit.usage import get_meter, merge_usage_into_meta

    llm_fn = getattr(mem, "_llm_fn", None)
    if not callable(llm_fn):
        return []
    try:
        qs = suggest_questions(mem.backend, llm_fn)
    except Exception:
        logger.exception("refresh_store_suggestions failed")
        qs = []
    usage = get_meter().take()
    try:
        if qs:
            store.update_meta(suggested_questions=qs)
        if usage.total_tokens:
            meta = store.meta()
            totals = merge_usage_into_meta(meta, usage, llm_spec)
            store.update_meta(usage_totals=totals)
    except Exception:
        logger.exception("failed to persist suggested_questions")
    return qs
