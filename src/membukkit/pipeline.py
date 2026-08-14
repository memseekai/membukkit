"""MemorySystem — the core MEMBUKKIT pipeline.

    mem = MemorySystem.from_pretrained(...)
    mem.ingest(sessions, dates)
    result = mem.answer(question, question_date="2024/01/01")

Fact storage lives behind a `MemoryBackend` (in-memory by default, Turbopuffer
in the service). `MemorySystem` owns the models (encoder/reranker), the query
router, the readers, and the cross-encoder rerank + RRF fusion; the backend
owns storage and candidate generation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from membukkit import telemetry
from membukkit.config import ModelConfig, PromptConfig, RetrievalConfig, StorageConfig
from membukkit.time_utils import (
    DateLike,
    datetime_sort_key,
    format_prompt_date,
    parse_datetime,
    to_iso8601,
)
from membukkit.usage import (
    estimate_cost_usd,
    estimate_tokens,
    get_meter,
    llm_model_spec,
    window_fraction,
)

logger = logging.getLogger(__name__)


def _source_ref(session_idx: int, turn_idx, n_turns: int) -> str:
    """Provenance pointer for a fact distilled from `session_idx`.

    The distiller backpointers each atomic fact to the [T{i}]-numbered turn it
    came from; when that index is valid we keep it (turn-level provenance),
    otherwise (idx=-1 fallback, or an out-of-range hallucination) the ref
    degrades to session granularity.
    """
    if isinstance(turn_idx, int) and 0 <= turn_idx < n_turns:
        return f"session:{session_idx}/turn:{turn_idx}"
    return f"session:{session_idx}"


# Metrics are fetched at call time (telemetry caches real instruments once
# configure() has run; before that it returns a no-op without caching).


@dataclass
class RetrievalTrace:
    """Inspectable trace of a retrieval operation."""

    opened_buckets: List[Dict] = field(default_factory=list)
    scan_fraction: float = 0.0
    n_facts: int = 0
    n_scanned: int = 0
    k_total: int = 0
    reader_type: str = "dated"
    ranked_facts: List[str] = field(default_factory=list)
    ranked_fact_times: List[Optional[str]] = field(default_factory=list)
    # Approximate tokens sent to the reader (memory lines + question, chars/4).
    est_reader_tokens: int = 0
    # backend / service observability (populated by the Turbopuffer backend)
    backend: str = "memory"
    perf: Dict = field(default_factory=dict)
    # Per-lane breakdown under union retrieval: {"verbatim": {...}, "atomic":
    # {...}} with each lane's opened buckets / scan_frac / exclusions, so the
    # activation trace stays unambiguous when bucket ids are lane-local.
    lanes: Dict = field(default_factory=dict)
    usage: Optional[Dict] = None
    est_cost_usd: Optional[float] = None
    window_fraction: float = 0.0
    model: str = ""


@dataclass
class AnswerResult:
    """Result of mem.answer()."""

    answer: Optional[str] = None
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)
    facts: List[str] = field(default_factory=list)


@dataclass
class MemorySearchHit:
    """One grounded memory fact returned by `MemorySystem.search`."""

    ref: str
    text: str
    fact: str
    timestamp: Optional[str] = None
    source_id: str = ""
    doc_id: str = ""
    doc_name: str = ""
    source_ref: str = ""
    kind: str = ""  # "verbatim" | "atomic" | ""
    status: str = "current"  # current | superseded | historical
    superseded_by: str = ""


@dataclass
class MemorySearchResult:
    """Evidence-only retrieval result, with no reader LLM generation."""

    query: str
    hits: List[MemorySearchHit] = field(default_factory=list)
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)


class MemorySystem:
    """Stateful memory system: ingest conversations, answer questions.

    The pipeline:
      1. ingest: distill turns -> atomic facts -> embed -> store/partition
      2. answer: route query -> candidate pool -> rerank -> fuse via RRF -> read
    """

    def __init__(
        self,
        encoder,
        reranker,
        llm_fn: Callable[[str], str],
        retrieval: RetrievalConfig,
        prompts: PromptConfig,
        distiller=None,
        backend=None,
        storage: Optional[StorageConfig] = None,
    ):
        self._encoder = encoder
        self._reranker = reranker
        self._llm_fn = llm_fn
        self._retrieval = retrieval
        self._prompts = prompts
        self._distiller = distiller

        if backend is None:
            from membukkit.storage import make_backend

            backend = make_backend(retrieval, encoder, storage)
        self._backend = backend

    @classmethod
    def from_pretrained(
        cls,
        models: Optional[ModelConfig] = None,
        retrieval: Optional[RetrievalConfig] = None,
        llm: str = "openai:gpt-4o-mini",
        prompts: Optional[PromptConfig] = None,
        storage: Optional[StorageConfig] = None,
    ) -> "MemorySystem":
        """Create a MemorySystem from pretrained models.

        Args:
            models: Model paths configuration. Uses defaults if None.
            retrieval: Retrieval configuration. Uses defaults if None.
            llm: LLM spec string like "openai:gpt-4o-mini" or "anthropic:claude-...".
            prompts: Prompt configuration. Uses defaults if None.
            storage: Storage backend configuration. Defaults to in-memory.
        """
        models = models or ModelConfig()
        retrieval = retrieval or RetrievalConfig()
        prompts = prompts or PromptConfig.default()

        from membukkit.models.registry import resolve_encoder_path, resolve_reranker_path
        from membukkit.models.encoder import Encoder
        from membukkit.models.reranker import UtilityReranker
        from membukkit.llm.backends import parse_llm_spec

        encoder = Encoder(resolve_encoder_path(models))
        reranker = UtilityReranker.load(resolve_reranker_path(models), device=models.device)
        llm_fn = parse_llm_spec(llm)

        from membukkit.extraction.distiller import FactDistiller

        distiller = FactDistiller(llm_fn, prompts=prompts)

        return cls(encoder, reranker, llm_fn, retrieval, prompts, distiller, storage=storage)

    @property
    def backend(self):
        """The storage backend (for persistence and inspection surfaces)."""
        return self._backend

    def reset(self) -> None:
        """Drop all stored facts (used between eval instances)."""
        self._backend.clear()

    @property
    def prompts(self) -> PromptConfig:
        return self._prompts

    def set_prompts(self, prompts: PromptConfig) -> None:
        """Replace prompt config (and sync the distiller) for this system."""
        self._prompts = prompts or PromptConfig.default()
        if self._distiller is not None:
            self._distiller.prompts = self._prompts

    def ingest(
        self,
        sessions: List[List[Dict[str, str]]],
        dates: Optional[Sequence[DateLike]] = None,
        subject: Optional[str] = None,
        doc_id: str = "",
        doc_name: str = "",
        doc_type: str = "",
        on_progress: Optional[Callable] = None,
    ):
        """Ingest conversation sessions into memory.

        Returns a :class:`~membukkit.reports.WriteReport` (int-compatible via
        ``int(report)`` / ``report.n_stored``).
        """
        from membukkit.extraction.distiller import build_transcript, make_key
        from membukkit.progress import emit
        from membukkit.reports import WriteReport
        from membukkit.retrieval.bucket_index import extract_entities, time_bucket_key
        from membukkit.storage.base import FactRecord, content_id
        from membukkit.supersession import link_supersessions

        if subject and self._distiller:
            self._distiller.subject = subject

        get_meter().take()  # clear prior LLM usage for this operation
        dates = dates or [None] * len(sessions)
        records: List[FactRecord] = []
        warnings: List[str] = []
        had_input = False

        # Pre-count non-empty sessions so distill bars have a stable total.
        nonempty = []
        for s_idx, session in enumerate(sessions):
            turns = [(t.get("role", "user"), t.get("content", "")) for t in session]
            if build_transcript(turns, numbered=True).strip():
                nonempty.append(s_idx)
        n_sessions = len(nonempty)
        detail_base = doc_name or doc_id or ""
        if n_sessions:
            emit(on_progress, "distill", 0, n_sessions, detail=detail_base)

        with telemetry.span("memory.ingest", n_sessions=len(sessions)) as sp:
            done_sessions = 0
            for s_idx, session in enumerate(sessions):
                turns = [(t.get("role", "user"), t.get("content", "")) for t in session]
                transcript = build_transcript(turns, numbered=True)
                if not transcript.strip():
                    continue
                had_input = True
                date_value = dates[s_idx] if s_idx < len(dates) else None
                ts = parse_datetime(date_value)
                date_str = format_prompt_date(ts)
                # Provenance must identify the ingested session globally, not
                # just its position in this call: a bare `ingest:{s_idx}` made
                # every standalone `add` claim to be "ingest:0", so unrelated
                # facts shared a source and anything reasoning about source
                # membership (erasure, drill-down) could act on the wrong rows.
                session_id = f"ingest:{s_idx}:{content_id(transcript, date=ts)[:8]}"
                # Prompt selection: file-ingested content (doc_type set) with
                # no assistant turn is not a user↔assistant chat — documents,
                # records, and multi-speaker exports need the subject-agnostic
                # prompt (the chat prompt extracts "USER facts" and returns
                # NONE for them). Passing mode only in document mode keeps
                # duck-typed distillers with the legacy 3-arg signature working.
                doc_mode = bool(doc_type) and not any(r == "assistant" for r, _ in turns)

                def _distill(key: str):
                    if doc_mode:
                        return self._distiller.distill(
                            key, transcript, date_str, mode="document"
                        )
                    return self._distiller.distill(key, transcript, date_str)

                if self._retrieval.union:
                    # Verbatim lane: EVERY non-empty turn (all roles), matching
                    # the eval's raw bank. Atomic lane: distilled facts (if a
                    # distiller is configured). Both are dated by the session.
                    for t_idx, turn in enumerate(session):
                        text = (turn.get("content", "") or "").strip()
                        if not text:
                            continue
                        records.append(
                            FactRecord(
                                text=text,
                                timestamp=ts,
                                source_session=session_id,
                                source_speaker=turn.get("role", "user"),
                                subject=subject,
                                entities=sorted(extract_entities(text)),
                                time_bucket=time_bucket_key(ts),
                                kind="verbatim",
                                doc_id=doc_id,
                                doc_name=doc_name,
                                source_ref=f"session:{s_idx}/turn:{t_idx}",
                            )
                        )
                    if self._distiller:
                        key = make_key("ingest", s_idx, transcript)
                        distilled = list(_distill(key))
                        if not distilled:
                            warnings.append(
                                f"empty_extract session={s_idx} (distiller returned no facts)"
                            )
                        for turn_idx, text in distilled:
                            records.append(
                                FactRecord(
                                    text=text,
                                    timestamp=ts,
                                    source_session=session_id,
                                    subject=subject,
                                    entities=sorted(extract_entities(text)),
                                    time_bucket=time_bucket_key(ts),
                                    kind="atomic",
                                    doc_id=doc_id,
                                    doc_name=doc_name,
                                    source_ref=_source_ref(s_idx, turn_idx, len(session)),
                                )
                            )
                    done_sessions += 1
                    emit(
                        on_progress,
                        "distill",
                        done_sessions,
                        n_sessions,
                        detail=detail_base or f"session {done_sessions}/{n_sessions}",
                    )
                    continue

                if self._distiller:
                    key = make_key("ingest", s_idx, transcript)
                    fact_items = list(_distill(key))
                    if not fact_items:
                        warnings.append(
                            f"empty_extract session={s_idx} (distiller returned no facts)"
                        )
                else:
                    # No distiller: store raw turns, but only what the person
                    # said — assistant/system turns are boilerplate, not facts
                    # about the subject.
                    fact_items = [
                        (t_idx, (turn.get("content", "") or "").strip())
                        for t_idx, turn in enumerate(session)
                        if turn.get("role", "user") not in ("assistant", "system")
                        and (turn.get("content", "") or "").strip()
                    ]

                for turn_idx, text in fact_items:
                    records.append(
                        FactRecord(
                            text=text,
                            timestamp=ts,
                            source_session=session_id,
                            subject=subject,
                            entities=sorted(extract_entities(text)),
                            time_bucket=time_bucket_key(ts),
                            doc_id=doc_id,
                            doc_name=doc_name,
                            source_ref=_source_ref(s_idx, turn_idx, len(session)),
                        )
                    )
                done_sessions += 1
                emit(
                    on_progress,
                    "distill",
                    done_sessions,
                    n_sessions,
                    detail=detail_base or f"session {done_sessions}/{n_sessions}",
                )

            n_verbatim = sum(1 for r in records if r.kind == "verbatim")
            n_extracted = sum(1 for r in records if r.kind != "verbatim")
            new_atomic_ids = [r.ensure_id() for r in records if r.kind == "atomic"]

            if records:
                try:
                    n_new = self._backend.upsert_facts(records, on_progress=on_progress)
                except TypeError:
                    # Duck-typed backends that only accept facts=.
                    n_new = self._backend.upsert_facts(records)
            else:
                n_new = 0

            superseded: List[Dict[str, str]] = []
            if new_atomic_ids:
                try:
                    superseded = link_supersessions(self._backend, new_atomic_ids)
                except Exception as e:
                    logger.debug("supersession link skipped: %s", e)
                    warnings.append(f"supersession_link_failed: {e}")

            status = "ok"
            if had_input and n_new == 0 and not records:
                status = "empty_extract"
                warnings.append("no facts stored from non-empty input")
            elif had_input and n_extracted == 0 and self._distiller and n_verbatim == 0:
                status = "empty_extract"
            elif not had_input and n_new == 0:
                status = "noop"

            usage = get_meter().take()
            model = llm_model_spec(self._llm_fn)
            if usage.total_tokens == 0 and self._distiller and had_input:
                # Cache-only distill: fall back to rough estimate from transcripts.
                est_chars = 0
                for s_idx in nonempty:
                    turns = [
                        (t.get("role", "user"), t.get("content", ""))
                        for t in sessions[s_idx]
                    ]
                    est_chars += len(build_transcript(turns, numbered=True))
                from membukkit.usage import TokenUsage

                usage = TokenUsage(
                    prompt_tokens=max(0, est_chars // 4),
                    completion_tokens=0,
                    source="estimate",
                    calls=0,
                )
            report = WriteReport(
                n_stored=int(n_new),
                n_extracted=n_extracted,
                n_verbatim=n_verbatim,
                superseded=superseded,
                warnings=warnings,
                status=status,
                usage=usage.to_dict() if usage.total_tokens or usage.calls else None,
                est_cost_usd=estimate_cost_usd(usage, model),
                model=model,
            )
            telemetry.set_attributes(
                sp, n_records=len(records), n_new=n_new, status=status
            )
            telemetry.counter("membukkit.facts.ingested").add(n_new)
            return report

    def delete_facts(
        self,
        fact_ids: Sequence[str],
        *,
        purge_source: bool = False,
    ) -> Dict[str, Any]:
        """Erase specific facts. Irreversible, and it rewrites history.

        MemBukkit's default model is append-and-supersede: a knowledge update
        marks the old fact superseded and keeps it, which is what makes as-of
        answers possible. This is the deliberate exception, for facts that are
        simply wrong or that the user does not want retained. An as-of question
        that used to return a deleted fact will no longer return it.

        Two things happen beyond dropping the named rows:

        - **The verbatim source goes too.** An atomic fact and the turn it was
          distilled from are separate rows, so removing only the fact would
          leave the same content in the verbatim lane, still retrievable. The
          source row is removed once no surviving fact still points at it.
        - **Supersession is repaired.** A fact that was superseded by one of
          the deleted facts becomes current again, which is what you want when
          the thing you are deleting was a bad correction.

        Args:
            fact_ids: Ids to erase (``MemorySearchHit.ref`` / evidence ``ref``).
            purge_source: Also erase everything else distilled from the same
                ingested turns. Use when the source content itself is the
                problem, rather than one fact drawn from it.

        Returns:
            ``{"deleted": int, "requested": int, "revived": [ids], "kinds": {...}}``
        """
        backend = self._backend
        requested = [f for f in dict.fromkeys(fact_ids) if f]
        if not requested:
            return {"deleted": 0, "requested": 0, "revived": [], "kinds": {}, "unknown": []}

        # Accept the `mem:<prefix>` refs that search and receipts print, not
        # just full stored ids.
        unknown: List[str] = []
        if hasattr(backend, "resolve_ids"):
            ids, unknown = backend.resolve_ids(requested)
        else:
            ids = requested
        if not ids:
            return {
                "deleted": 0,
                "requested": len(requested),
                "revived": [],
                "kinds": {},
                "unknown": unknown,
            }

        if purge_source and hasattr(backend, "source_group_ids"):
            targets = list(dict.fromkeys([*ids, *backend.source_group_ids(ids)]))
        elif hasattr(backend, "orphaned_source_ids"):
            targets = list(dict.fromkeys([*ids, *backend.orphaned_source_ids(ids)]))
        else:
            targets = ids

        target_set = set(targets)
        revived = []
        kinds: Dict[str, int] = {}
        if hasattr(backend, "list_rows_for_ids"):
            for row in backend.list_rows_for_ids(targets):
                kinds[row.get("kind", "")] = kinds.get(row.get("kind", ""), 0) + 1
        if hasattr(backend, "list_atomic_rows"):
            revived = [
                r["id"]
                for r in backend.list_atomic_rows()
                if r.get("superseded_by") in target_set and r["id"] not in target_set
            ]

        n = backend.delete_facts(targets)
        telemetry.counter("membukkit.facts.deleted").add(n)
        return {
            "deleted": n,
            "requested": len(requested),
            "revived": revived,
            "kinds": kinds,
            "unknown": unknown,
        }

    def forget(
        self,
        *,
        doc_id: str = "",
        source_session: str = "",
    ) -> Dict[str, Any]:
        """Erase everything ingested from one source. Irreversible.

        The bulk counterpart to :meth:`delete_facts`: removes both lanes for a
        whole document or conversation, so "forget that file" is one call
        rather than a fact-by-fact sweep.
        """
        backend = self._backend
        if not (doc_id or source_session):
            raise ValueError("forget() needs doc_id or source_session")
        if not hasattr(backend, "ids_for_source"):
            raise NotImplementedError(
                f"{type(backend).__name__} does not support forget(); "
                "delete_facts() with explicit ids still works."
            )
        ids = backend.ids_for_source(doc_id=doc_id, source_session=source_session)
        return self.delete_facts(ids)

    def ingest_facts(
        self,
        facts: Sequence[Any],
        subject: Optional[str] = None,
    ) -> int:
        """Ingest pre-normalized facts directly, bypassing the LLM distiller.

        Use this for structured sources (calendar events, mined tool results)
        where the fact text is already atomic and dated — the distiller adds no
        value and would only add cost and latency.

        Args:
            facts: An iterable of fact items. Each item may be a mapping or any
                   object exposing these fields (mapping keys or attributes):
                     - ``text``      (required) the fact string
                     - ``timestamp`` or ``date`` the fact's time (datetime/date/
                       ISO8601/legacy string); optional
                     - ``source``    source-session label for provenance; optional
                     - ``fact_id`` or ``id`` a stable id *seed*. When present the
                       row dedupes on this seed (scoped by subject) instead of on
                       content alone — so two calendar instances with identical
                       text but different ids are kept as distinct facts.
            subject: Optional name of the person whose memory this is.

        Returns:
            The number of *new* facts written.
        """
        from membukkit.retrieval.bucket_index import extract_entities, time_bucket_key
        from membukkit.storage.base import FactRecord, content_id

        def _field(item, *names):
            for n in names:
                if isinstance(item, dict):
                    if n in item and item[n] is not None:
                        return item[n]
                elif getattr(item, n, None) is not None:
                    return getattr(item, n)
            return None

        records: List[FactRecord] = []
        for item in facts:
            text = _field(item, "text")
            text = (text or "").strip() if isinstance(text, str) else ""
            if not text:
                continue
            ts = parse_datetime(_field(item, "timestamp", "date"))
            source = _field(item, "source", "source_session")
            seed = _field(item, "fact_id", "id")

            rec = FactRecord(
                text=text,
                timestamp=ts,
                source_session=source if isinstance(source, str) else None,
                subject=subject,
                entities=sorted(extract_entities(text)),
                time_bucket=time_bucket_key(ts),
                doc_id=str(_field(item, "doc_id") or ""),
                doc_name=str(_field(item, "doc_name") or ""),
                source_ref=str(_field(item, "source_ref") or ""),
            )
            if seed:
                # Source-aware stable id: dedupe on the seed, not the text, so
                # repeated/recurring instances are not collapsed by content.
                rec.id = content_id(str(seed), subject)
            records.append(rec)

        with telemetry.span("memory.ingest_facts", n_facts=len(records)) as sp:
            n_new = self._backend.upsert_facts(records) if records else 0
            telemetry.set_attributes(sp, n_records=len(records), n_new=n_new)
            telemetry.counter("membukkit.facts.ingested").add(n_new)
            return n_new

    def partition(self) -> Dict:
        """Build or return the topic partition (centroids + buckets)."""
        return self._backend.partition()

    def label_buckets(
        self,
        llm: Optional[Callable] = None,
        kind: Optional[str] = None,
        on_progress: Optional[Callable] = None,
    ) -> Dict[int, str]:
        """Auto-label topic buckets by asking an LLM to summarize exemplars.

        With `kind`, buckets are that lane's own (lane-local partition) —
        matching what lane-scoped surfaces like the GUI memory map display.
        Transient per-bucket LLM failures are skipped (partial labels are
        still useful); if EVERY bucket fails, the error is raised so callers
        can surface it instead of caching junk placeholder labels.
        """
        from membukkit.progress import emit

        if kind is None:
            part = self.partition()
            k_eff = part.get("k_eff", 0) if part else 0
        else:
            view = self._backend.lane_view(kind)
            k_eff = view.get("k_eff", 0) if view else 0
        if k_eff < 1:
            return {}
        llm = llm or self._llm_fn
        labels: Dict[int, str] = {}
        last_err: Optional[Exception] = None
        emit(on_progress, "label", 0, k_eff)
        for b in range(k_eff):
            exemplars = (
                self._backend.topic_exemplars(b, n=5, kind=kind)
                if kind
                else self._backend.topic_exemplars(b, n=5)
            )
            if not exemplars:
                emit(on_progress, "label", b + 1, k_eff, detail=f"bucket {b}")
                continue
            prompt = (
                "These facts belong to the same topic cluster. "
                "Give a SHORT label (2-5 words) for this topic:\n"
                + "\n".join(f"- {e}" for e in exemplars)
                + "\n\nLabel:"
            )
            try:
                labels[b] = llm(prompt).strip()
            except Exception as e:
                last_err = e
                logger.warning("labeling bucket %d failed: %s", b, e)
            emit(on_progress, "label", b + 1, k_eff, detail=f"bucket {b}")
        if not labels and last_err is not None:
            raise RuntimeError(f"bucket labeling failed: {last_err}") from last_err
        return labels

    @staticmethod
    def _filter_cands(
        cands: List,
        as_of: Optional[datetime],
        include_history: bool,
    ) -> List:
        from membukkit.supersession import is_active_as_of

        out = []
        for c in cands:
            if is_active_as_of(
                superseded_by=getattr(c, "superseded_by", "") or "",
                valid_to=getattr(c, "valid_to", None),
                timestamp=getattr(c, "timestamp", None),
                as_of=as_of,
                include_history=include_history,
            ):
                out.append(c)
        return out

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        exclude_buckets: Optional[Dict[str, Sequence[int]]] = None,
        question_date: DateLike = None,
        include_history: bool = False,
    ) -> MemorySearchResult:
        """Retrieve dated memory evidence without asking the reader LLM.

        This is the evidence surface for agentic/deep-research workflows: it
        uses the same routing, backend candidate generation, cross-encoder
        reranking, and temporal presentation as `answer()`, but returns stable
        citation refs and trace metadata instead of a natural-language answer.

        By default, superseded facts are hidden (active-as-of ``question_date``
        or live-only when no date). Pass ``include_history=True`` to keep them
        with a ``status`` badge on each hit.
        """
        if self._backend.count() == 0:
            return MemorySearchResult(query=query, trace=RetrievalTrace(reader_type="search"))

        from membukkit.retrieval.router import (
            is_recommendation_query,
            is_reasoning_query,
            is_temporal_query,
        )
        from membukkit.supersession import fact_status

        is_rec = is_recommendation_query(query)
        is_reason = (not is_rec) and is_reasoning_query(query)
        is_temp = is_temporal_query(query)
        as_of = parse_datetime(question_date)

        k_eff = top_k or self._retrieval.top_k
        if top_k is None and self._retrieval.reasoning_top_k > k_eff and is_reason:
            k_eff = self._retrieval.reasoning_top_k

        with telemetry.timed(
            "memory.search",
            telemetry.histogram("membukkit.search.duration"),
            is_temporal=is_temp,
            is_reason=is_reason,
            top_k=k_eff,
        ) as sp:
            v_cands, a_cands, trace_dict = self._retrieve_lanes(
                query, k_eff, is_reason, is_temp, is_rec, exclude_buckets
            )
            v_cands = self._filter_cands(v_cands, as_of, include_history)
            a_cands = self._filter_cands(a_cands, as_of, include_history)
            # Deep-research surface: a single globally-sorted evidence list keeps
            # each hit aligned with its citation ref (unique across lanes via id).
            cands = list(v_cands) + list(a_cands)
            fact_lines, fact_times = self._present_temporal_with_times(cands)
            ordered_cands = self._order_temporal(cands)

            hits: List[MemorySearchHit] = []
            for idx, (cand, line, ts) in enumerate(
                zip(ordered_cands, fact_lines, fact_times), start=1
            ):
                source_id = getattr(cand, "id", "") or ""
                ref_seed = source_id[:12] if source_id else str(idx)
                sb = getattr(cand, "superseded_by", "") or ""
                hits.append(
                    MemorySearchHit(
                        ref=f"mem:{ref_seed}",
                        text=getattr(cand, "text", ""),
                        fact=line,
                        timestamp=ts,
                        source_id=source_id,
                        doc_id=getattr(cand, "doc_id", ""),
                        doc_name=getattr(cand, "doc_name", ""),
                        source_ref=getattr(cand, "source_ref", ""),
                        kind=getattr(cand, "kind", "") or "",
                        status=fact_status(
                            superseded_by=sb,
                            valid_to=getattr(cand, "valid_to", None),
                            timestamp=getattr(cand, "timestamp", None),
                            as_of=as_of,
                        ),
                        superseded_by=sb,
                    )
                )

            trace = self._trace_from_dict(
                trace_dict,
                reader_type="search",
                fact_lines=fact_lines,
                fact_times=fact_times,
                question=query,
            )
            telemetry.set_attributes(
                sp,
                backend=trace.backend,
                scan_fraction=trace.scan_fraction,
                n_facts=trace.n_facts,
                n_scanned=trace.n_scanned,
                est_reader_tokens=trace.est_reader_tokens,
            )
            return MemorySearchResult(query=query, hits=hits, trace=trace)

    def answer(
        self,
        question: str,
        question_date: DateLike = None,
        identity: str = "",
        generate_answer: bool = True,
        exclude_buckets: Optional[Dict[str, Sequence[int]]] = None,
        include_history: bool = False,
    ) -> AnswerResult:
        """Answer a question from memory.

        Returns an AnswerResult with .answer, .trace, .facts. When
        ``generate_answer`` is False, the reader LLM is skipped entirely (no
        cost) and ``.answer`` is None — retrieval still runs, so ``.facts`` is
        populated for a facts-only query.

        ``exclude_buckets`` closes topic buckets for this call — the control
        interface for topic-scoped exclusion and interventions. It maps a lane
        kind ("verbatim"/"atomic", or "single" for non-union) to lane-local
        bucket ids; excluded buckets can never contribute evidence.
        """
        if self._backend.count() == 0:
            return AnswerResult(answer="N/I" if generate_answer else None, trace=RetrievalTrace())

        from membukkit.retrieval.router import (
            is_changelog_query,
            is_recommendation_query,
            is_reasoning_query,
            is_temporal_query,
        )
        from membukkit.reading.readers import (
            make_dated_reader,
            make_recommendation_reader,
            make_reasoning_reader,
            _normalize_abstain,
        )

        is_rec = is_recommendation_query(question)
        reasoning = is_reasoning_query(question)
        is_reason = (not is_rec) and reasoning  # reader selection is gated by is_rec
        is_temp = is_temporal_query(question)
        is_changelog = is_changelog_query(question)

        # The reasoning top_k bump keys off the RAW reasoning signal (matching the
        # eval), independent of the recommendation gate used for reader choice.
        k_eff = self._retrieval.top_k
        if self._retrieval.reasoning_top_k > k_eff and reasoning:
            k_eff = self._retrieval.reasoning_top_k

        get_meter().take()  # clear prior LLM usage for this ask
        with telemetry.timed(
            "memory.answer",
            telemetry.histogram("membukkit.answer.duration"),
            is_temporal=is_temp,
            is_reason=is_reason,
        ) as sp:
            try:
                v_cands, a_cands, trace_dict = self._retrieve_lanes(
                    question, k_eff, is_reason, is_temp, is_rec, exclude_buckets
                )
                as_of = parse_datetime(question_date)
                v_cands = self._filter_cands(v_cands, as_of, include_history)
                a_cands = self._filter_cands(a_cands, as_of, include_history)
                # Present each lane chronologically, then concatenate verbatim
                # then atomic (NOT a single global sort) — exact `coremem_union`.
                v_lines, v_times = self._present_temporal_with_times(v_cands)
                a_lines, a_times = self._present_temporal_with_times(a_cands)
                fact_lines = v_lines + a_lines
                fact_times = v_times + a_times
                prompt_date = format_prompt_date(question_date)

                reader_type = "rec" if is_rec else ("reason" if is_reason else "dated")

                if generate_answer:
                    from membukkit.prompts.resolve import resolve_reader_template

                    if is_rec:
                        reader = make_recommendation_reader(
                            self._llm_fn,
                            identity=identity,
                            prompt_template=resolve_reader_template(
                                self._prompts, "recommendation"
                            ),
                        )
                    elif is_reason:
                        reader = make_reasoning_reader(
                            self._llm_fn,
                            identity=identity,
                            prompt_template=resolve_reader_template(
                                self._prompts,
                                "reasoning",
                                changelog=is_changelog,
                            ),
                        )
                    else:
                        reader = make_dated_reader(
                            self._llm_fn,
                            identity=identity,
                            prompt_template=resolve_reader_template(
                                self._prompts,
                                "dated",
                                changelog=is_changelog,
                            ),
                        )

                    try:
                        ans = reader(fact_lines, question, prompt_date)
                    except Exception:
                        ans = "N/I"
                    ans = _normalize_abstain(ans)
                else:
                    ans = None
            except Exception as e:
                telemetry.counter("membukkit.errors").add(1, {"stage": "answer"})
                sp.record_exception(e)
                raise

            mode = trace_dict.get("mode", "")
            trace = self._trace_from_dict(
                trace_dict,
                reader_type=reader_type,
                fact_lines=fact_lines,
                fact_times=fact_times,
                question=question,
            )
            usage = get_meter().take()
            model = llm_model_spec(self._llm_fn)
            if usage.total_tokens == 0 and generate_answer and trace.est_reader_tokens:
                from membukkit.usage import TokenUsage

                usage = TokenUsage(
                    prompt_tokens=trace.est_reader_tokens,
                    completion_tokens=estimate_tokens(ans or ""),
                    source="estimate",
                    calls=1 if ans else 0,
                )
            trace.usage = usage.to_dict() if usage.total_tokens or usage.calls else None
            trace.est_cost_usd = estimate_cost_usd(usage, model)
            trace.window_fraction = window_fraction(trace.est_reader_tokens)
            trace.model = model
            telemetry.set_attributes(
                sp,
                reader_type=reader_type,
                mode=mode,
                backend=trace.backend,
                scan_fraction=trace.scan_fraction,
                n_facts=trace.n_facts,
                est_reader_tokens=trace.est_reader_tokens,
                n_scanned=trace.n_scanned,
                k_total=trace.k_total,
            )
            telemetry.counter("membukkit.queries").add(
                1, {"reader_type": reader_type, "mode": mode, "backend": trace.backend}
            )
            telemetry.histogram("membukkit.scan_fraction", unit="1").record(
                trace.scan_fraction, {"mode": mode} if mode else {}
            )
            # Union cost/latency: when both lanes ran, surface the additive DB
            # server time so the ~2x union overhead is measurable in dashboards.
            perf = trace.perf or {}
            per_lane = perf.get("per_lane")
            if per_lane:
                union_server_ms = float(perf.get("server_total_ms") or 0.0)
                telemetry.set_attributes(
                    sp,
                    union_lanes=len(per_lane),
                    union_server_ms=union_server_ms,
                    union_bytes_queried=int(perf.get("bytes_queried") or 0),
                )
                telemetry.histogram("membukkit.union.db.duration").record(
                    union_server_ms, {"reader_type": reader_type}
                )
            if telemetry.capture_content() and ans is not None:
                telemetry.set_attributes(sp, question=question, answer_text=ans)
            return AnswerResult(answer=ans, trace=trace, facts=fact_lines)

    def _retrieve_lanes(
        self,
        query: str,
        top_k: int,
        is_reason: bool,
        is_temporal: bool,
        is_rec: bool,
        exclude_buckets: Optional[Dict[str, Sequence[int]]] = None,
    ) -> Tuple[List, List, Dict]:
        """Retrieve the union lanes: returns (verbatim_cands, atomic_cands, trace).

        Reproduces `coremem_union`: the verbatim lane honours `is_temporal`; the
        atomic lane is always non-temporal; recommendation queries skip atomic.
        Non-union config returns (cands, [], trace) — single-index behaviour.
        `exclude_buckets` maps a lane kind ("verbatim"/"atomic"; "single" for
        the non-union index) to lane-local topic bucket ids to close.
        """
        cfg = self._retrieval
        excl = exclude_buckets or {}
        if not cfg.union:
            cands, trace = self._retrieve(
                query, top_k, is_reason, is_temporal, exclude_buckets=excl.get("single")
            )
            return cands, [], trace

        lanes = cfg.union_lanes or ("verbatim", "atomic")
        want_v = "verbatim" in lanes
        want_a = "atomic" in lanes
        # Recommendation routing only applies to the dual-lane union: atomic
        # facts dilute the preference signal, so route to verbatim only.
        if want_v and want_a and is_rec and cfg.union_recommendation_verbatim_only:
            want_a = False

        v_cands: List = []
        v_trace: Optional[Dict] = None
        if want_v:
            t0 = time.perf_counter()
            v_cands, v_trace = self._retrieve(
                query,
                top_k,
                is_reason,
                is_temporal,
                kind="verbatim",
                exclude_buckets=excl.get("verbatim"),
            )
            if v_trace is not None:
                v_trace.setdefault("perf", {})["lane_ms"] = (time.perf_counter() - t0) * 1000.0

        a_cands: List = []
        a_trace: Optional[Dict] = None
        if want_a and self._backend.count_kind("atomic") > 0:
            t0 = time.perf_counter()
            a_cands, a_trace = self._retrieve(
                query,
                top_k,
                is_reason,
                False,
                kind="atomic",
                exclude_buckets=excl.get("atomic"),
            )
            if a_trace is not None:
                a_trace.setdefault("perf", {})["lane_ms"] = (time.perf_counter() - t0) * 1000.0

        if v_trace and a_trace:
            trace = self._merge_union_trace(v_trace, a_trace)
        else:
            trace = v_trace or a_trace or {}
        lanes_out = {}
        for lane_name, lane_trace in (("verbatim", v_trace), ("atomic", a_trace)):
            if lane_trace is None:
                continue
            lanes_out[lane_name] = {
                k: lane_trace[k]
                for k in (
                    "buckets",
                    "scan_frac",
                    "n_facts",
                    "n_scanned",
                    "k_total",
                    "excluded_buckets",
                    "n_excluded",
                )
                if k in lane_trace
            }
        if lanes_out and isinstance(trace, dict):
            trace["lanes"] = lanes_out
        return v_cands, a_cands, trace

    @staticmethod
    def _merge_union_trace(v: Dict, a: Optional[Dict]) -> Dict:
        """Merge two lanes' traces. Cost/latency is additive (the union runs two
        retrievals), so we SUM the server-side perf and keep a per-lane breakdown
        rather than dropping the atomic lane's numbers."""
        if not a:
            return v
        return {
            "buckets": (v.get("buckets") or v.get("topic_buckets") or [])
            + (a.get("buckets") or a.get("topic_buckets") or []),
            "scan_frac": max(v.get("scan_frac", 0.0), a.get("scan_frac", 0.0)),
            "n_facts": v.get("n_facts", 0) + a.get("n_facts", 0),
            "n_scanned": v.get("n_scanned", 0) + a.get("n_scanned", 0),
            "k_total": v.get("k_total", 0),
            "backend": v.get("backend", "memory"),
            "perf": MemorySystem._merge_perf(v.get("perf", {}), a.get("perf", {})),
            "mode": v.get("mode", ""),
        }

    @staticmethod
    def _merge_perf(v: Dict, a: Dict) -> Dict:
        """Combine two lanes' Turbopuffer perf blocks: sum the additive cost/latency
        numbers, keep a per-kind breakdown so the union's overhead is visible."""

        def _num(d: Dict, *keys) -> float:
            for k in keys:
                val = d.get(k)
                if isinstance(val, (int, float)):
                    return float(val)
            return 0.0

        server_ms = _num(v, "server_total_ms", "query_execution_ms") + _num(
            a, "server_total_ms", "query_execution_ms"
        )
        bytes_q = _num(v, "billable_logical_bytes_queried", "bytes_queried") + _num(
            a, "billable_logical_bytes_queried", "bytes_queried"
        )
        merged: Dict = {
            "server_total_ms": server_ms,
            "bytes_queried": bytes_q,
            "lane_ms": _num(v, "lane_ms") + _num(a, "lane_ms"),
            "per_lane": {
                "verbatim": {
                    "lane_ms": _num(v, "lane_ms"),
                    "server_ms": _num(v, "server_total_ms"),
                },
                "atomic": {"lane_ms": _num(a, "lane_ms"), "server_ms": _num(a, "server_total_ms")},
            },
        }
        return merged

    def _retrieve(
        self,
        query: str,
        top_k: int,
        is_reason: bool = False,
        is_temporal: bool = False,
        kind: Optional[str] = None,
        exclude_buckets: Optional[Sequence[int]] = None,
    ) -> Tuple[List, Dict]:
        """Candidate generation (backend) -> cross-encoder rerank -> top_k.

        `kind` scopes retrieval to one union lane (verbatim/atomic); None keeps
        the single-index behaviour. `exclude_buckets` closes topic buckets for
        this call (lane-local ids; topic mode only).
        """
        cfg = self._retrieval
        with telemetry.timed(
            "memory.retrieve",
            telemetry.histogram("membukkit.retrieve.duration"),
            top_k=top_k,
            kind=kind or "single",
        ) as sp:
            pool = self._backend.candidates(
                query,
                top_k=top_k,
                is_reason=is_reason,
                is_temporal=is_temporal,
                kind=kind,
                exclude_buckets=exclude_buckets,
            )
            cands = pool.candidates
            mode = pool.trace.get("mode", "")
            telemetry.set_attributes(
                sp, pool_size=len(cands), has_cosine=pool.has_cosine, mode=mode
            )
            telemetry.histogram("membukkit.pool_size", unit="1").record(
                len(cands), {"mode": mode} if mode else {}
            )
            if not cands:
                return [], pool.trace

            with telemetry.timed(
                "memory.rerank",
                telemetry.histogram("membukkit.rerank.duration"),
                n_candidates=len(cands),
            ):
                util = self._reranker.score(query, [c.text for c in cands])

            if cfg.select == "hybrid" and pool.has_cosine:
                from membukkit.retrieval.buckets import rrf_order

                cos = np.asarray([c.cosine for c in cands], dtype=np.float64)
                if pool.has_lexical:
                    lex = np.asarray([c.lexical for c in cands], dtype=np.float64)
                    order = rrf_order(util, cos, lex, k_rrf=cfg.k_rrf)
                else:
                    order = rrf_order(util, cos, k_rrf=cfg.k_rrf)
            else:
                order = list(np.argsort(util)[::-1])

            sel = order[:top_k]
            return [cands[j] for j in sel], pool.trace

    @staticmethod
    def _present_temporal(cands: List) -> List[str]:
        """Present candidate facts in chronological order with a date prefix."""
        return MemorySystem._present_temporal_with_times(cands)[0]

    @staticmethod
    def _present_temporal_with_times(cands: List) -> Tuple[List[str], List[Optional[str]]]:
        """Present candidate facts in chronological order plus ISO timestamps."""
        pairs = []
        for c in cands:
            t = getattr(c, "timestamp", None)
            ds = t.strftime("%Y-%m-%d") if t is not None else "unknown-date"
            pairs.append((datetime_sort_key(t), f"[{ds}] {c.text}", to_iso8601(t)))
        ordered = sorted(pairs, key=lambda x: x[0])
        return [ln for _, ln, _ in ordered], [ts for _, _, ts in ordered]

    @staticmethod
    def _order_temporal(cands: List) -> List:
        """Return candidates in the same chronological order used for display."""
        return [
            c
            for _, c in sorted(
                ((datetime_sort_key(getattr(c, "timestamp", None)), c) for c in cands),
                key=lambda x: x[0],
            )
        ]

    def _trace_from_dict(
        self,
        trace_dict: Dict,
        *,
        reader_type: str,
        fact_lines: List[str],
        fact_times: List[Optional[str]],
        question: str = "",
    ) -> RetrievalTrace:
        memory_block = "\n".join(fact_lines)
        return RetrievalTrace(
            opened_buckets=trace_dict.get("buckets", trace_dict.get("topic_buckets", [])),
            scan_fraction=trace_dict.get("scan_frac", 0.0),
            n_facts=trace_dict.get("n_facts", self._backend.count()),
            n_scanned=trace_dict.get("n_scanned", 0),
            k_total=trace_dict.get("k_total", 0),
            reader_type=reader_type,
            ranked_facts=fact_lines,
            ranked_fact_times=fact_times,
            est_reader_tokens=estimate_tokens(memory_block, question),
            backend=trace_dict.get("backend", "memory"),
            perf=trace_dict.get("perf", {}),
            lanes=trace_dict.get("lanes", {}),
        )

    @staticmethod
    def _parse_date(date_str: DateLike) -> Optional[datetime]:
        return parse_datetime(date_str)
