"""Shared plumbing for the human-facing CLI commands."""

from __future__ import annotations

import sys
from datetime import date
from typing import Any, Optional, Tuple

from membukkit.config import ModelConfig, PromptConfig, RetrievalConfig
from membukkit.pipeline import MemorySystem
from membukkit.progress import ProgressEvent
from membukkit.storage.localstore import LocalStore, stores_root

DEFAULT_LLM = "openai:gpt-4o-mini"
DEFAULT_ENCODER = "biencoder_v1"  # resolved via registry (HF Hub / fallback)


def empty_store_hint(name: str) -> str:
    """Actionable message when a store exists but has no facts."""
    return (
        f"store {name!r} is empty — add something first:\n"
        f"  membukkit add \"…\" --store {name}\n"
        f"  membukkit ingest ./notes --store {name}\n"
        f"  membukkit ui --demo personal-assistant"
    )


def resolve_as_of(
    mem: MemorySystem,
    explicit: Optional[str] = None,
    *,
    announce: bool = True,
) -> str:
    """Pick an as-of date: explicit → latest fact date → today.

    Matches the GUI default for non-demo stores (latest fact date when present).
    """
    raw = (explicit or "").strip()
    if raw:
        return raw
    latest_fn = getattr(mem.backend, "latest_fact_date", None)
    latest = latest_fn() if callable(latest_fn) else None
    if latest:
        if announce:
            print(
                f"[as-of {latest} — latest fact date; pass --as-of YYYY-MM-DD to override]",
                file=sys.stderr,
            )
        return str(latest)
    today = date.today().isoformat()
    if announce:
        print(
            f"[as-of {today} — today; pass --as-of YYYY-MM-DD to override]",
            file=sys.stderr,
        )
    return today


def make_tqdm_progress(desc: str = ""):
    """Return ``(on_progress, close)`` driving a live tqdm bar when possible.

    Quiet (no-op callback) when stdout is not a TTY or tqdm is missing — CI-
    friendly. The bar retargets when the phase or total changes.
    """
    state: dict[str, Any] = {"bar": None, "phase": None, "total": None}

    def close() -> None:
        bar = state["bar"]
        if bar is not None:
            bar.close()
            state["bar"] = None

    if not sys.stdout.isatty():
        return (lambda _ev: None), close

    try:
        from tqdm.auto import tqdm
    except Exception:
        return (lambda _ev: None), close

    def on_progress(ev: ProgressEvent) -> None:
        phase = ev.phase or desc or "work"
        if state["bar"] is None or state["phase"] != phase or state["total"] != ev.total:
            close()
            if ev.total <= 0:
                return
            state["bar"] = tqdm(
                total=ev.total,
                desc=phase,
                dynamic_ncols=True,
                unit="it",
                leave=True,
            )
            state["phase"] = phase
            state["total"] = ev.total
        bar = state["bar"]
        if bar is None:
            return
        # tqdm wants absolute n; clamp in case callers jump ahead.
        bar.n = min(max(ev.done, 0), bar.total or ev.done)
        if ev.detail:
            bar.set_postfix_str(ev.detail[:48], refresh=False)
        bar.refresh()

    return on_progress, close


def build_encoder(spec: str):
    """Encoder from a spec: 'openai:MODEL[@DIMS]' or a local/HF model path."""
    if spec.startswith("openai:"):
        from membukkit.models.openai_encoder import make_openai_encoder

        return make_openai_encoder(spec)
    from membukkit.models.encoder import Encoder
    from membukkit.models.registry import resolve_encoder_path

    return Encoder(resolve_encoder_path(ModelConfig(encoder=spec)))


def build_system(
    llm: str = DEFAULT_LLM,
    encoder_spec: str = DEFAULT_ENCODER,
    distill: bool = True,
    retrieval: Optional[RetrievalConfig] = None,
    prompts: Optional[PromptConfig] = None,
) -> MemorySystem:
    """A MemorySystem on the in-memory backend with the given providers."""
    from membukkit.llm.backends import parse_llm_spec
    from membukkit.models.registry import resolve_reranker_path
    from membukkit.models.reranker import UtilityReranker

    prompts = prompts or PromptConfig.default()
    encoder = build_encoder(encoder_spec)
    reranker = UtilityReranker.load(resolve_reranker_path(ModelConfig()))
    llm_fn = parse_llm_spec(llm)
    distiller = None
    if distill:
        from membukkit.extraction.distiller import FactDistiller

        distiller = FactDistiller(llm_fn, prompts=prompts)
    return MemorySystem(
        encoder=encoder,
        reranker=reranker,
        llm_fn=llm_fn,
        retrieval=retrieval or RetrievalConfig(),
        prompts=prompts,
        distiller=distiller,
    )


def resolve_prompts_arg(prompt_pack: Optional[str] = None) -> PromptConfig:
    """CLI helper: load a pack id/path or return defaults."""
    if not prompt_pack:
        return PromptConfig.default()
    from membukkit.prompts.packs import load_prompt_pack

    try:
        return load_prompt_pack(prompt_pack)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e


def open_store(
    name: str,
    llm: str = DEFAULT_LLM,
    encoder_spec: Optional[str] = None,
    distill: bool = True,
    create: bool = False,
    prompts: Optional[PromptConfig] = None,
    prompt_pack: Optional[str] = None,
) -> Tuple[MemorySystem, LocalStore]:
    """Open (or create) a named store and load it into a MemorySystem.

    The encoder spec is pinned in the store's metadata at creation so later
    sessions never mix embedding spaces; passing a different spec for an
    existing store is an error unless it matches.

    Prompt priority: explicit ``prompts`` → ``prompt_pack`` CLI/pack →
    store-persisted ``prompts`` in meta → defaults.
    """
    try:
        store = LocalStore(name, create=create)
    except FileNotFoundError:
        raise SystemExit(
            f"store {name!r} not found under {stores_root()}\n"
            f"  list stores:  membukkit stores\n"
            f"  create one:   membukkit add \"…\" --store {name}\n"
            f"  or try a demo: membukkit ui --demo personal-assistant"
        ) from None
    meta = store.meta()
    pinned = meta.get("encoder")
    if encoder_spec and pinned and encoder_spec != pinned:
        raise SystemExit(
            f"store {name!r} was built with encoder {pinned!r}; "
            f"cannot open it with {encoder_spec!r}"
        )
    spec = encoder_spec or pinned or DEFAULT_ENCODER
    if not pinned:
        store.update_meta(encoder=spec)

    if prompts is None and prompt_pack:
        prompts = resolve_prompts_arg(prompt_pack)
    if prompts is None and meta.get("prompts"):
        prompts = PromptConfig.from_dict(meta.get("prompts"))
    mem = build_system(llm=llm, encoder_spec=spec, distill=distill, prompts=prompts)
    n = store.load_backend(mem.backend)
    if n:
        print(f"[store {name!r}: {n} facts loaded]", file=sys.stderr)
    _autoscale_budget(mem)
    return mem, store


# Below this many facts, bucket routing has nothing to save — scan everything.
# The scan budget is a large-corpus efficiency lever, not a small-store one.
_FULL_SCAN_BELOW = 500


def _autoscale_budget(mem: MemorySystem) -> None:
    if mem.backend.count() < _FULL_SCAN_BELOW:
        cfg = mem._retrieval
        cfg.scan_budget = 1.0
        cfg.scan_budget_reason = 1.0
        cfg.scan_budget_temporal = 1.0


def distill_store(mem: MemorySystem, store: LocalStore, on_progress=None) -> int:
    """Re-run fact extraction over a store's preserved raw documents.

    Rescues stores that were ingested verbatim-only (``--no-distill``, or a
    build that predates distillation). Idempotent: fact ids hash on
    text+date+kind, so existing verbatim/atomic facts dedupe on upsert and
    only the missing distilled facts are added.

    Returns the number of new facts written.
    """
    from membukkit.progress import ProgressEvent, emit

    if mem._distiller is None:
        raise RuntimeError("no distiller configured (LLM unavailable?)")
    work = []
    for doc in store.documents():
        content = store.document_content(doc["doc_id"])
        if not content or not content.get("sessions"):
            continue
        work.append((doc, content))
    n_new = 0
    n_docs = len(work)
    for di, (doc, content) in enumerate(work):
        name = doc.get("name", "") or doc["doc_id"]

        def _forward(ev: ProgressEvent, _name=name):
            detail = ev.detail or _name
            if _name and _name not in detail:
                detail = f"{_name} · {detail}" if detail else _name
            on_progress(ProgressEvent(ev.phase, ev.done, ev.total, detail))

        if on_progress is not None and n_docs > 1:
            emit(
                on_progress,
                "distill",
                di,
                n_docs,
                detail=f"document {di + 1}/{n_docs}: {name}",
            )
        report = mem.ingest(
            content["sessions"],
            dates=content.get("dates"),
            doc_id=doc["doc_id"],
            doc_name=doc.get("name", ""),
            doc_type=doc.get("type", "document"),
            on_progress=_forward if on_progress is not None else None,
        )
        n_new += int(report)
    if n_new:
        store.save_backend(mem.backend)
    return n_new


def format_write_receipt(report, *, llm: str = "") -> str:
    """One-line write receipt with tokens / $."""
    from membukkit.usage import TokenUsage, format_cost, format_usage_line

    model = getattr(report, "model", None) or llm
    usage = TokenUsage.from_dict(getattr(report, "usage", None))
    cost = getattr(report, "est_cost_usd", None)
    parts = [
        f"status={report.status}",
        f"stored={report.n_stored}",
        f"superseded={len(report.superseded)}",
    ]
    if usage.total_tokens or usage.calls:
        parts.append(format_usage_line(usage, model))
    elif cost is not None:
        parts.append(format_cost(cost))
    return "  ".join(parts)


def record_store_usage(store: LocalStore, report_or_trace, model: str = "") -> None:
    """Accumulate operation usage into store meta."""
    from membukkit.usage import TokenUsage, merge_usage_into_meta

    usage_dict = getattr(report_or_trace, "usage", None)
    if isinstance(report_or_trace, dict):
        usage_dict = report_or_trace.get("usage")
    if not usage_dict:
        return
    model = model or getattr(report_or_trace, "model", "") or ""
    meta = store.meta()
    totals = merge_usage_into_meta(meta, TokenUsage.from_dict(usage_dict), model)
    store.update_meta(usage_totals=totals)


def format_answer(result, show_trace: bool = False, *, as_of: str = "") -> str:
    """Human-readable answer + always-on receipt; optional deep trace."""
    from membukkit.usage import TokenUsage, format_cost, window_fraction

    lines = [result.answer or "(no answer)"]
    t = result.trace
    est = getattr(t, "est_reader_tokens", 0) or 0
    usage = TokenUsage.from_dict(getattr(t, "usage", None))
    cost = getattr(t, "est_cost_usd", None)
    wf = getattr(t, "window_fraction", None)
    if wf is None:
        wf = window_fraction(est)
    tag = usage.source if usage.total_tokens else "est."
    receipt = (
        f"--- receipt: as-of {as_of or '?'} · ~{est:,} reader tokens "
        f"({wf * 100:.2f}% of 128k) · scanned {t.scan_fraction:.0%} "
        f"({t.n_scanned}/{t.n_facts} facts)"
    )
    if cost is not None:
        receipt += f" · {format_cost(cost)} ({tag})"
    elif usage.total_tokens:
        receipt += (
            f" · {usage.prompt_tokens:,} in / {usage.completion_tokens:,} out ({tag})"
        )
    lines.append("")
    lines.append(receipt)
    if show_trace:
        lines.append(
            f"--- trace: reader={t.reader_type}"
            + (f", model={t.model}" if getattr(t, "model", None) else "")
        )
        for lane, info in (t.lanes or {}).items():
            raw = info.get("buckets", [])
            buckets = ",".join(
                str(b.get("bucket") if isinstance(b, dict) else b) for b in raw
            )
            lines.append(f"    {lane}: buckets [{buckets}] scan {info.get('scan_frac', 0):.0%}")
        lines.append("--- memories used:")
        for fact in t.ranked_facts[:15]:
            lines.append(f"    {fact[:160]}")
    return "\n".join(lines)
