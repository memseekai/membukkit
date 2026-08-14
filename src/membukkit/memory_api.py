"""Mem0-shaped facade: ``Memory.add`` / ``search`` / ``ask``.

Thin wrapper over :class:`~membukkit.pipeline.MemorySystem` so agent builders
don't need sessions/dates/doc_type on day one. Power users keep using
``MemorySystem`` directly.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Union

from membukkit.config import ModelConfig, PromptConfig, RetrievalConfig, StorageConfig
from membukkit.pipeline import MemorySearchResult, MemorySystem
from membukkit.reports import AskReceipt, EvidenceItem, WriteReport

Message = Dict[str, str]
AddInput = Union[str, Message, Sequence[Message], Sequence[Sequence[Message]]]


def _as_sessions(content: AddInput) -> List[List[Message]]:
    if isinstance(content, str):
        text = content.strip()
        if not text:
            return []
        return [[{"role": "user", "content": text}]]
    if isinstance(content, dict):
        role = content.get("role", "user")
        text = (content.get("content") or "").strip()
        if not text:
            return []
        return [[{"role": role, "content": text}]]
    if not content:
        return []
    # List of messages (one session) vs list of sessions.
    first = content[0]
    if isinstance(first, dict):
        return [list(content)]  # type: ignore[arg-type]
    return [list(s) for s in content]  # type: ignore[arg-type]


class Memory:
    """Agent-friendly long-term memory with receipts."""

    def __init__(self, system: MemorySystem):
        self._sys = system

    @classmethod
    def from_pretrained(
        cls,
        models: Optional[ModelConfig] = None,
        retrieval: Optional[RetrievalConfig] = None,
        llm: str = "openai:gpt-4o-mini",
        prompts: Optional[PromptConfig] = None,
        storage: Optional[StorageConfig] = None,
    ) -> "Memory":
        return cls(
            MemorySystem.from_pretrained(
                models=models,
                retrieval=retrieval,
                llm=llm,
                prompts=prompts,
                storage=storage,
            )
        )

    @classmethod
    def wrap(cls, system: MemorySystem) -> "Memory":
        return cls(system)

    @property
    def system(self) -> MemorySystem:
        return self._sys

    @property
    def backend(self):
        return self._sys.backend

    def add(
        self,
        content: AddInput,
        *,
        subject: str = "",
        date: Optional[Union[str, datetime, date]] = None,
        doc_id: str = "",
        doc_name: str = "",
        on_progress=None,
    ) -> WriteReport:
        """Store a message, transcript, or session list. Returns a write receipt.

        ``subject`` attributes the distilled facts to a person, so the extractor
        writes in their voice and does not mix up other people's details.
        """
        sessions = _as_sessions(content)
        if not sessions:
            return WriteReport(status="noop", warnings=["empty content"])
        dates = [date] * len(sessions) if date is not None else None
        return self._sys.ingest(
            sessions,
            dates=dates,
            subject=subject or None,
            doc_id=doc_id,
            doc_name=doc_name or (f"subject:{subject}" if subject else ""),
            doc_type="chat",
            on_progress=on_progress,
        )

    def delete(self, *fact_ids: str, purge_source: bool = False) -> Dict[str, Any]:
        """Erase facts the user does not want kept. Irreversible.

        Memory is append-and-supersede by default: updates mark the old fact
        superseded and keep it, which is what makes as-of answers work. Use
        this for facts that are wrong, or that should not be retained at all.

            mem.delete(r.evidence[0].ref)

        The verbatim turn a fact came from is removed with it (unless another
        surviving fact still needs it), and anything the deleted fact had
        superseded becomes current again. Pass ``purge_source=True`` to also
        erase everything else distilled from the same turns.
        """
        return self._sys.delete_facts(list(fact_ids), purge_source=purge_source)

    def forget(self, *, doc_id: str = "", source_session: str = "") -> Dict[str, Any]:
        """Erase everything from one document or conversation. Irreversible."""
        return self._sys.forget(doc_id=doc_id, source_session=source_session)

    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        as_of: Optional[Union[str, datetime, date]] = None,
        include_history: bool = False,
    ) -> MemorySearchResult:
        """Retrieve evidence with scan-budget trace (no reader LLM).

        Retrieval is not scoped by subject: one store is one memory. Keep
        separate people in separate stores.
        """
        return self._sys.search(
            query,
            top_k=top_k,
            question_date=as_of,
            include_history=include_history,
        )

    def ask(
        self,
        query: str,
        *,
        as_of: Optional[Union[str, datetime, date]] = None,
        include_history: bool = False,
        top_k: Optional[int] = None,
    ) -> AskReceipt:
        """Answer with dated memory + clickable-style evidence receipts."""
        as_of_eff = as_of or date.today().isoformat()
        # Reader uses active-as-of facts; evidence always keeps history so
        # receipts can badge superseded facts (Mem0-style stale-fact wound).
        search = self._sys.search(
            query,
            top_k=top_k,
            question_date=as_of_eff,
            include_history=True,
        )
        result = self._sys.answer(
            query,
            question_date=as_of_eff,
            include_history=include_history,
        )
        t = result.trace
        evidence = [
            EvidenceItem(
                ref=h.ref,
                fact=h.fact,
                text=h.text,
                timestamp=h.timestamp,
                fact_id=h.source_id,
                doc_id=h.doc_id,
                doc_name=h.doc_name,
                source_ref=h.source_ref,
                kind=h.kind,
                status=h.status,
                superseded_by=h.superseded_by,
            )
            for h in search.hits
        ]
        return AskReceipt(
            answer=result.answer,
            scan_fraction=t.scan_fraction,
            n_facts=t.n_facts,
            n_scanned=t.n_scanned,
            est_reader_tokens=t.est_reader_tokens,
            reader_type=t.reader_type,
            question_date=str(as_of_eff),
            evidence=evidence,
            lanes=t.lanes or {},
            opened_buckets=list(t.opened_buckets or []),
            usage=getattr(t, "usage", None),
            est_cost_usd=getattr(t, "est_cost_usd", None),
            window_fraction=float(getattr(t, "window_fraction", 0) or 0),
            model=getattr(t, "model", "") or "",
        )
