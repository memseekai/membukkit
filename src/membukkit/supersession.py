"""Lightweight supersession linking for mutable atomic facts.

Marks older facts as superseded when a newer, highly similar atomic fact
arrives — no Neo4j, no silent DELETE. Turbopuffer already exposes
``backend.supersede``; the in-memory backend gets the same contract.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Soft signal that a fact is about mutable state (not required, but boosts match).
_MUTABLE = re.compile(
    r"\b("
    r"rent|lease|salary|wage|lives?\s+in|moved\s+to|relocated|"
    r"employer|works?\s+at|job|role|married|divorced|prefer|"
    r"switched\s+to|raised|became|changed|now\s+|currently|"
    r"notify|notification|breach|liability|cap|timeout"
    r")\b",
    re.I,
)


def _naive(ts: Optional[datetime]) -> Optional[datetime]:
    if ts is None:
        return None
    return ts.replace(tzinfo=None) if getattr(ts, "tzinfo", None) else ts


def fact_status(
    *,
    superseded_by: str = "",
    valid_to: Optional[datetime] = None,
    timestamp: Optional[datetime] = None,
    as_of: Optional[datetime] = None,
) -> str:
    """Return ``current`` / ``superseded`` / ``historical`` for receipts."""
    as_of_n = _naive(as_of)
    ts_n = _naive(timestamp)
    vt_n = _naive(valid_to)
    if as_of_n is not None and ts_n is not None and ts_n > as_of_n:
        return "historical"
    if superseded_by:
        if as_of_n is None:
            return "superseded"
        if vt_n is None or vt_n <= as_of_n:
            return "superseded"
    return "current"


def is_active_as_of(
    *,
    superseded_by: str = "",
    valid_to: Optional[datetime] = None,
    timestamp: Optional[datetime] = None,
    as_of: Optional[datetime] = None,
    include_history: bool = False,
) -> bool:
    """Whether a fact should enter the evidence pool."""
    as_of_n = _naive(as_of)
    ts_n = _naive(timestamp)
    vt_n = _naive(valid_to)
    if as_of_n is not None and ts_n is not None and ts_n > as_of_n:
        return False
    if include_history:
        return True
    if as_of_n is None:
        return not superseded_by
    if vt_n is not None and vt_n <= as_of_n:
        return False
    if superseded_by and vt_n is None:
        return False
    return True


def link_supersessions(
    backend,
    new_ids: Sequence[str],
    *,
    threshold: float = 0.78,
) -> List[Dict[str, str]]:
    """Mark older atomic facts superseded by newer similar ones.

    Returns list of ``{old_id, new_id}`` pairs that were linked.
    """
    if not new_ids or not hasattr(backend, "list_atomic_rows"):
        return []
    rows = backend.list_atomic_rows()
    if len(rows) < 2:
        return []
    by_id = {r["id"]: r for r in rows}
    new_set = [fid for fid in new_ids if fid in by_id]
    if not new_set:
        return []

    # Need embeddings for cosine; fall back to entity+keyword overlap.
    embs = getattr(backend, "_embs", None)
    id_to_idx = {fid: i for i, fid in enumerate(getattr(backend, "_ids", []))}

    pairs: List[Tuple[str, str, Optional[datetime]]] = []
    for new_id in new_set:
        new = by_id[new_id]
        new_ts = _naive(new.get("timestamp"))
        new_ents = set(new.get("entities") or [])
        new_mutable = bool(_MUTABLE.search(new.get("text") or ""))
        best_old = None
        best_score = threshold
        for old in rows:
            old_id = old["id"]
            if old_id == new_id or old_id in new_set:
                continue
            if old.get("superseded_by"):
                continue
            old_ts = _naive(old.get("timestamp"))
            if new_ts is not None and old_ts is not None and not (new_ts > old_ts):
                continue
            if new_ts is not None and old_ts is None:
                continue
            score = 0.0
            ni, oi = id_to_idx.get(new_id), id_to_idx.get(old_id)
            if embs is not None and ni is not None and oi is not None:
                a = np.asarray(embs[ni], dtype=np.float32)
                b = np.asarray(embs[oi], dtype=np.float32)
                denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
                score = float(a @ b) / denom
            else:
                old_ents = set(old.get("entities") or [])
                overlap = len(new_ents & old_ents)
                if overlap:
                    score = 0.55 + 0.1 * overlap
                if new_mutable and _MUTABLE.search(old.get("text") or ""):
                    score = max(score, 0.72)
            # Require some entity or mutable signal unless cosine is strong.
            if score < 0.88:
                old_ents = set(old.get("entities") or [])
                if not (new_ents & old_ents) and not (
                    new_mutable and _MUTABLE.search(old.get("text") or "")
                ):
                    continue
            if score >= best_score:
                best_score = score
                best_old = old_id
        if best_old:
            pairs.append((best_old, new_id, new_ts))

    if not pairs:
        return []

    # Dedup: one old fact linked at most once (keep newest new_id).
    by_old: Dict[str, Tuple[str, Optional[datetime]]] = {}
    for old_id, new_id, when in pairs:
        prev = by_old.get(old_id)
        if prev is None:
            by_old[old_id] = (new_id, when)
            continue
        prev_when = _naive(by_id[prev[0]].get("timestamp"))
        if when is not None and (prev_when is None or when >= prev_when):
            by_old[old_id] = (new_id, when)

    final = [(o, n, w) for o, (n, w) in by_old.items()]
    if hasattr(backend, "supersede"):
        backend.supersede([(o, n) for o, n, _ in final], when=None)
        # Patch valid_to per-pair when timestamps differ.
        if hasattr(backend, "set_valid_to"):
            for old_id, new_id, when in final:
                if when is not None:
                    backend.set_valid_to(old_id, when)

    return [{"old_id": o, "new_id": n} for o, n, _ in final]
