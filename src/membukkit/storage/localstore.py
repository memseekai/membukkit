"""Local persistent stores: named, on-disk memory banks with raw sources.

A store lives under ``~/.membukkit/stores/<name>/`` (override the root with
``MEMBUKKIT_HOME``):

    facts.jsonl    one row per stored fact (text, dates, kind, provenance)
    vectors.npy    embedding matrix aligned with facts.jsonl rows
    docs.jsonl     registry of ingested source documents
    sources/       raw source content, one JSON file per document
    meta.json      store metadata (encoder spec, timestamps, counts)

This is what makes ``membukkit ingest`` / ``ask`` / the GUI work across
processes without a cloud vector DB: ingest once, query forever. The raw
sources are kept so any fact can be traced back to the exact passage it came
from (``FactRecord.doc_id`` + ``source_ref``).
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def stores_root() -> Path:
    home = os.environ.get("MEMBUKKIT_HOME")
    root = Path(home).expanduser() if home else Path.home() / ".membukkit"
    return root / "stores"


def list_stores() -> List[Dict]:
    """All local stores with their metadata."""
    root = stores_root()
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        meta_path = d / "meta.json"
        if not d.is_dir() or not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = {}
        meta["name"] = d.name
        out.append(meta)
    return out


_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

# Lexical turn attribution for legacy atomic facts whose source_ref is
# session-only. Deliberately deterministic and model-free: content words
# shared with fewer turns count for more, so the fact's distinctive terms
# (names, numbers, rare nouns) dominate over conversational filler.
_STOPWORDS = frozenset(
    """a an the and or but if then so of to in on at for with from by as is are
    was were be been being am it its this that these those i you he she we
    they my your his her our their me him them what which who when where how
    not no do does did done have has had will would can could should may
    might there here about into over after before all any some more most
    other just also very really said says told""".split()
)
_WORD_RE = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> set:
    return {
        t
        for t in _WORD_RE.findall((text or "").lower())
        if len(t) > 2 and t not in _STOPWORDS
    }


def best_matching_turn(fact_text: str, turns: List[Dict]) -> Optional[int]:
    """Index of the turn that best matches `fact_text` by lexical overlap.

    Scores each turn by the fact's content tokens it contains, weighted by
    inverse document frequency across the session's turns. Returns None when
    nothing overlaps (the caller then shows the session unhighlighted).
    Ties resolve to the earliest turn, so results are stable.
    """
    fact_tokens = _content_tokens(fact_text)
    if not fact_tokens or not turns:
        return None
    per_turn = [_content_tokens(t.get("content", "")) for t in turns]
    df: Dict[str, int] = {}
    for toks in per_turn:
        for tok in fact_tokens & toks:
            df[tok] = df.get(tok, 0) + 1
    best_idx: Optional[int] = None
    best_score = 0.0
    for i, toks in enumerate(per_turn):
        score = sum(1.0 / df[tok] for tok in fact_tokens & toks)
        if score > best_score + 1e-9:
            best_idx, best_score = i, score
    return best_idx


class LocalStore:
    """One named on-disk store. Cheap to construct; loads lazily."""

    def __init__(self, name: str, create: bool = True):
        if not _NAME_RE.match(name):
            raise ValueError(
                f"invalid store name {name!r} (use letters, digits, '-', '_')"
            )
        self.name = name
        self.dir = stores_root() / name
        if create:
            (self.dir / "sources").mkdir(parents=True, exist_ok=True)
            if not self.meta_path.exists():
                self._write_meta({"created_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        elif not self.dir.exists():
            raise FileNotFoundError(f"store {name!r} not found under {stores_root()}")

    # ------------------------------------------------------------------ paths
    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def facts_path(self) -> Path:
        return self.dir / "facts.jsonl"

    @property
    def vectors_path(self) -> Path:
        return self.dir / "vectors.npy"

    @property
    def docs_path(self) -> Path:
        return self.dir / "docs.jsonl"

    # ------------------------------------------------------------------- meta
    def meta(self) -> Dict:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text())
        return {}

    def _write_meta(self, meta: Dict) -> None:
        self.meta_path.write_text(json.dumps(meta, indent=1))

    def update_meta(self, **kv) -> None:
        meta = self.meta()
        meta.update(kv)
        self._write_meta(meta)

    # --------------------------------------------------------- backend state
    def save_backend(self, backend) -> None:
        """Persist an InMemoryBackend's facts + vectors."""
        state = backend.to_state()
        with open(self.facts_path, "w") as f:
            for row in state["facts"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if state["vectors"] is not None:
            np.save(self.vectors_path, np.asarray(state["vectors"], dtype=np.float32))
        elif self.vectors_path.exists():
            self.vectors_path.unlink()
        self.update_meta(
            n_facts=len(state["facts"]),
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    def load_backend(self, backend) -> int:
        """Restore facts + vectors into a fresh InMemoryBackend. Returns count."""
        if not self.facts_path.exists():
            return 0
        rows = [json.loads(line) for line in open(self.facts_path)]
        vectors = np.load(self.vectors_path) if self.vectors_path.exists() else None
        backend.from_state(rows, vectors)
        return len(rows)

    # ---------------------------------------------------------------- sources
    def add_document(
        self,
        name: str,
        sessions: List[List[Dict[str, str]]],
        dates: Optional[List[Optional[str]]] = None,
        doc_type: str = "document",
        origin: str = "",
    ) -> str:
        """Register a source document and keep its raw content.

        `sessions`/`dates` are the exact structures handed to
        ``MemorySystem.ingest`` so a ``source_ref`` like "session:3/turn:5"
        resolves unambiguously back to raw content.
        """
        doc_id = uuid.uuid4().hex[:12]
        record = {
            "doc_id": doc_id,
            "name": name,
            "type": doc_type,
            "origin": origin,
            "n_sessions": len(sessions),
            "added_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(self.docs_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        content = {"doc_id": doc_id, "name": name, "sessions": sessions, "dates": dates}
        (self.dir / "sources" / f"{doc_id}.json").write_text(
            json.dumps(content, ensure_ascii=False)
        )
        return doc_id

    def remove_document(self, doc_id: str) -> bool:
        """Drop a document from the registry and delete its raw source.

        Returns True when anything was removed. The caller is responsible for
        deleting the document's facts from the backend and re-saving it.
        """
        docs = self.documents()
        keep = [d for d in docs if d.get("doc_id") != doc_id]
        source_path = self.dir / "sources" / f"{doc_id}.json"
        if len(keep) == len(docs) and not source_path.exists():
            return False
        with open(self.docs_path, "w") as f:
            for d in keep:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        source_path.unlink(missing_ok=True)
        return True

    def documents(self) -> List[Dict]:
        if not self.docs_path.exists():
            return []
        return [json.loads(line) for line in open(self.docs_path)]

    def document_content(self, doc_id: str) -> Optional[Dict]:
        path = self.dir / "sources" / f"{doc_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    _REF_RE = re.compile(r"session:(\d+)(?:/turn:(\d+))?")

    def resolve_source(
        self,
        doc_id: str,
        source_ref: str,
        context: int = 2,
        fact_text: Optional[str] = None,
    ) -> Optional[Dict]:
        """Resolve a fact's provenance pointer to the raw source passage.

        Returns the referenced turn (or whole session) plus `context`
        surrounding turns, so a GUI can show the fact in situ.

        ``highlight_kind`` in the result says how the highlighted turn was
        chosen: ``"stored"`` when the ref itself carries a turn index, or
        ``"lexical"`` when the ref is session-only (legacy atomic facts) and
        the best-matching turn was picked by deterministic lexical overlap
        against ``fact_text``.
        """
        content = self.document_content(doc_id)
        if content is None:
            return None
        m = self._REF_RE.search(source_ref or "")
        if not m:
            return {"doc_id": doc_id, "name": content.get("name"), "excerpt": None}
        s_idx = int(m.group(1))
        sessions = content.get("sessions") or []
        if s_idx >= len(sessions):
            return None
        session = sessions[s_idx]
        dates = content.get("dates") or []
        date = dates[s_idx] if s_idx < len(dates) else None

        t_idx: Optional[int] = None
        highlight_kind: Optional[str] = None
        if m.group(2) is not None and int(m.group(2)) < len(session):
            t_idx = int(m.group(2))
            highlight_kind = "stored"
        elif fact_text:
            t_idx = best_matching_turn(fact_text, session)
            highlight_kind = "lexical" if t_idx is not None else None

        if t_idx is None:
            return {
                "doc_id": doc_id,
                "name": content.get("name"),
                "session": s_idx,
                "date": date,
                "turns": session,
                "highlight": None,
                "highlight_kind": None,
            }
        lo = max(0, t_idx - context)
        hi = min(len(session), t_idx + context + 1)
        return {
            "doc_id": doc_id,
            "name": content.get("name"),
            "session": s_idx,
            "date": date,
            "turns": session[lo:hi],
            "highlight": t_idx - lo,
            "highlight_kind": highlight_kind,
        }

    # ----------------------------------------------------------------- delete
    def delete(self) -> None:
        import shutil

        if self.dir.exists():
            shutil.rmtree(self.dir)
