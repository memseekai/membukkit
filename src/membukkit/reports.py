"""Write / ask report types for the agent-facing Memory API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WriteReport:
    """Receipt for an ingest / ``Memory.add`` call.

    Never looks like success when distillation produced nothing usable:
    ``status`` is ``"empty_extract"`` when the LLM (or verbatim path) stored
    zero facts from non-empty input.
    """

    n_stored: int = 0
    n_extracted: int = 0
    n_verbatim: int = 0
    superseded: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    status: str = "ok"  # ok | empty_extract | noop
    usage: Optional[Dict[str, Any]] = None
    est_cost_usd: Optional[float] = None
    model: str = ""

    def __int__(self) -> int:
        return self.n_stored

    def __bool__(self) -> bool:
        return self.n_stored > 0

    def __index__(self) -> int:
        return self.n_stored

    def __gt__(self, other: Any) -> bool:
        return self.n_stored > int(other)

    def __ge__(self, other: Any) -> bool:
        return self.n_stored >= int(other)

    def __lt__(self, other: Any) -> bool:
        return self.n_stored < int(other)

    def __le__(self, other: Any) -> bool:
        return self.n_stored <= int(other)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, WriteReport):
            return self.n_stored == other.n_stored and self.status == other.status
        if isinstance(other, int):
            return self.n_stored == other
        return NotImplemented

    def __add__(self, other: Any) -> int:
        if isinstance(other, WriteReport):
            return self.n_stored + other.n_stored
        return self.n_stored + int(other)

    def __radd__(self, other: Any) -> int:
        if other == 0:
            return self.n_stored
        return int(other) + self.n_stored

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_stored": self.n_stored,
            "n_extracted": self.n_extracted,
            "n_verbatim": self.n_verbatim,
            "superseded": list(self.superseded),
            "warnings": list(self.warnings),
            "status": self.status,
            "usage": self.usage,
            "est_cost_usd": self.est_cost_usd,
            "model": self.model,
        }


@dataclass
class EvidenceItem:
    """One cited memory with optional supersession badge."""

    ref: str
    fact: str
    text: str = ""
    timestamp: Optional[str] = None
    fact_id: str = ""
    doc_id: str = ""
    doc_name: str = ""
    source_ref: str = ""
    kind: str = ""
    status: str = "current"  # current | superseded | historical
    superseded_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref": self.ref,
            "fact": self.fact,
            "text": self.text,
            "timestamp": self.timestamp,
            "fact_id": self.fact_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "source_ref": self.source_ref,
            "kind": self.kind,
            "status": self.status,
            "superseded_by": self.superseded_by,
        }


@dataclass
class AskReceipt:
    """Answer + explainability surface for ``Memory.ask``."""

    answer: Optional[str]
    scan_fraction: float = 0.0
    n_facts: int = 0
    n_scanned: int = 0
    est_reader_tokens: int = 0
    reader_type: str = ""
    question_date: str = ""
    evidence: List[EvidenceItem] = field(default_factory=list)
    lanes: Dict[str, Any] = field(default_factory=dict)
    opened_buckets: List[Any] = field(default_factory=list)
    bucket_labels: Dict[str, str] = field(default_factory=dict)
    usage: Optional[Dict[str, Any]] = None
    est_cost_usd: Optional[float] = None
    window_fraction: float = 0.0
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "question_date": self.question_date,
            "scan_fraction": self.scan_fraction,
            "est_reader_tokens": self.est_reader_tokens,
            "usage": self.usage,
            "est_cost_usd": self.est_cost_usd,
            "window_fraction": self.window_fraction,
            "model": self.model,
            "trace": {
                "scan_fraction": self.scan_fraction,
                "n_facts": self.n_facts,
                "n_scanned": self.n_scanned,
                "est_reader_tokens": self.est_reader_tokens,
                "reader_type": self.reader_type,
                "lanes": self.lanes,
                "opened_buckets": self.opened_buckets,
                "bucket_labels": self.bucket_labels,
                "usage": self.usage,
                "est_cost_usd": self.est_cost_usd,
                "window_fraction": self.window_fraction,
                "model": self.model,
            },
            "evidence": [e.to_dict() for e in self.evidence],
        }
