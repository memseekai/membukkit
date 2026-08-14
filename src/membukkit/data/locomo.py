"""LoCoMo -> LongMemEval-instance adapter for MEMBUKKIT."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from membukkit.data.instance import LongMemEvalDataset, LongMemEvalInstance
from membukkit.time_utils import parse_datetime, to_iso8601

logger = logging.getLogger(__name__)

# Official LoCoMo data file from the snap-research release (verified live).
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
_LOCOMO_CACHE = Path.home() / ".cache" / "membukkit" / "locomo" / "locomo10.json"

CATEGORY_LABEL = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

CATEGORY_TO_LME_TASK = {
    1: "multi-session",
    2: "temporal-reasoning",
    3: "single-session-user",
    4: "single-session-user",
    5: "single-session-user",
}


def _parse_locomo_ts(ts_str: str) -> Optional[datetime]:
    """Parse LoCoMo timestamps, e.g. '1:56 pm on 8 May, 2023'."""
    if not ts_str:
        return None
    parsed = parse_datetime(ts_str)
    if parsed is not None:
        return parsed
    s = ts_str.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    m = re.match(r"(\d{1,2}:\d{2}\s*[ap]m)\s+on\s+(\d{1,2}\s+\w+,?\s+\d{4})", s, re.IGNORECASE)
    if m:
        time_part, date_part = m.group(1).strip(), m.group(2).strip()
        for dfmt in ("%d %B, %Y", "%d %B %Y"):
            for tfmt in ("%I:%M %p", "%I:%M%p"):
                try:
                    return datetime.strptime(f"{date_part} {time_part}", f"{dfmt} {tfmt}")
                except ValueError:
                    continue
    return None


def _norm_date(dt: Optional[datetime]) -> str:
    return to_iso8601(dt) or ""


def _split_evidence(evidence) -> set:
    out: set = set()
    if not evidence:
        return out
    items = evidence if isinstance(evidence, list) else [evidence]
    for e in items:
        for part in re.split(r"[;,]", str(e)):
            part = part.strip()
            if part:
                out.add(part)
    return out


class LoCoMoInstance(LongMemEvalInstance):
    """LongMemEval instance for LoCoMo.

    Differences from the base class:
      - `ability` is the LoCoMo category name.
      - `distill_scope_id` is the CONVERSATION id, so atomic-fact distillation
        cache is shared by all QA over the same conversation.
      - gold needles are computed lazily from evidence dia_ids via a shared
        dia->fact-index map.
    """

    @property
    def ability(self) -> str:
        return CATEGORY_LABEL.get(self._item.get("locomo_category", 1), "single_hop")

    @property
    def distill_scope_id(self) -> str:
        return self._item.get("_distill_scope", self.question_id)

    def get_gold_needle_indices(self, context_window: int = 0) -> List[int]:
        if self._facts is None:
            self._extract_facts()
        dia_map = self._item.get("_dia_to_fact_idx", {})
        n = len(self._facts or [])
        needles = sorted({dia_map[d] for d in self._item.get("_evidence", []) if d in dia_map})
        needles = [i for i in needles if 0 <= i < n]
        if not needles:
            return self.get_gold_fact_indices()
        if context_window <= 0:
            return needles
        fact_sessions = getattr(self, "_fact_session_idx", [])
        keep = set()
        for ni in needles:
            s = fact_sessions[ni] if ni < len(fact_sessions) else -1
            for j in range(ni - context_window, ni + context_window + 1):
                if 0 <= j < n and (j >= len(fact_sessions) or fact_sessions[j] == s):
                    keep.add(j)
        return sorted(keep)


def _http_get(url: str, timeout: float = 60.0) -> bytes:
    """Fetch a URL (separate function so tests can mock the HTTP call)."""
    from urllib.request import urlopen

    with urlopen(url, timeout=timeout) as resp:
        return resp.read()


def _download_locomo(cache_path: Path) -> Path:
    """Download locomo10.json from the official repo into the local cache."""
    logger.info(f"LoCoMo data not found locally — downloading {LOCOMO_URL} -> {cache_path}")
    try:
        raw = _http_get(LOCOMO_URL)
        json.loads(raw)  # never cache an HTML error page as data
    except Exception as e:
        raise FileNotFoundError(
            f"LoCoMo data not found locally and auto-download from {LOCOMO_URL} "
            f"failed ({type(e).__name__}: {e}). Download locomo10.json manually "
            f"(e.g. into {cache_path} or the current directory), or pass an "
            "explicit --locomo-path."
        ) from e
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_bytes(raw)
    tmp.replace(cache_path)
    logger.info(f"LoCoMo data cached at {cache_path}")
    return cache_path


def _conv_speakers(conv: Dict) -> List[str]:
    names = set()
    for key in ("speaker_a", "speaker_b"):
        if conv.get(key):
            names.add(conv[key])
    return sorted(names)


class _ConvHaystack:
    __slots__ = ("sessions", "dates", "session_ids", "qdate",
                 "dia_to_fact_idx", "dia_to_session_idx")

    def __init__(self, sessions, dates, session_ids, qdate, dia_to_fact_idx, dia_to_session_idx):
        self.sessions = sessions
        self.dates = dates
        self.session_ids = session_ids
        self.qdate = qdate
        self.dia_to_fact_idx = dia_to_fact_idx
        self.dia_to_session_idx = dia_to_session_idx


def _build_conv_haystack(conv: Dict) -> _ConvHaystack:
    container = conv.get("conversation", conv.get("sessions", {}))
    sessions: List[List[Dict]] = []
    dates: List[str] = []
    session_ids: List[str] = []
    dia_to_fact_idx: Dict[str, int] = {}
    dia_to_session_idx: Dict[str, int] = {}
    latest: Optional[datetime] = None
    fact_idx = 0

    idx = 1
    while True:
        key = f"session_{idx}"
        if key not in container:
            break
        dt = _parse_locomo_ts(container.get(f"{key}_date_time", ""))
        if dt and (latest is None or dt > latest):
            latest = dt
        s_idx = len(sessions)
        turns: List[Dict] = []
        for t in (container.get(key, []) or []):
            if not isinstance(t, dict):
                continue
            text = (t.get("text", "") or "").strip()
            if not text:
                continue
            speaker = t.get("speaker", "") or ""
            dia_id = t.get("dia_id", "")
            content = f"[{speaker}]: {text}" if speaker else text
            turns.append({"role": speaker or "user", "content": content})
            if dia_id:
                dia_to_fact_idx[dia_id] = fact_idx
                dia_to_session_idx[dia_id] = s_idx
            fact_idx += 1
        sessions.append(turns)
        dates.append(_norm_date(dt))
        session_ids.append(key)
        idx += 1

    return _ConvHaystack(sessions, dates, session_ids, _norm_date(latest),
                         dia_to_fact_idx, dia_to_session_idx)


def load_locomo(
    path: str,
    conversation_ids: Optional[List[int]] = None,
    max_instances: Optional[int] = None,
    drop_categories: Optional[List[int]] = None,
) -> LongMemEvalDataset:
    """Load LoCoMo as a LongMemEval-compatible dataset (one instance per QA pair).

    Args:
        path: path to locomo10.json (a list of 10 conversations).
        conversation_ids: 0-indexed conversations to load (None = all).
        max_instances: cap total instances (for smoke runs).
        drop_categories: LoCoMo categories to exclude (e.g. [5] to skip adversarial).
    """
    # Resolution order: explicit path, cwd fallback, local cache, then
    # auto-download from the official repo as a last resort.
    p = Path(path)
    if not p.exists():
        alt = Path("locomo10.json")
        if alt.exists():
            p = alt
        elif _LOCOMO_CACHE.exists():
            p = _LOCOMO_CACHE
        else:
            p = _download_locomo(_LOCOMO_CACHE)
    data = json.loads(p.read_text())
    if not isinstance(data, list):
        data = [data]

    conv_ids = conversation_ids if conversation_ids is not None else list(range(len(data)))
    drop = set(drop_categories or [])
    instances: List[LoCoMoInstance] = []

    for ci in conv_ids:
        if ci >= len(data):
            continue
        conv = data[ci]
        sample_id = conv.get("sample_id", f"conv{ci}")
        hs = _build_conv_haystack(conv)

        for qi, qa in enumerate(conv.get("qa", [])):
            cat = int(qa.get("category", 1))
            if cat in drop:
                continue
            evidence = sorted(_split_evidence(qa.get("evidence")))
            is_abs = cat == 5

            gold_sessions = {hs.dia_to_session_idx[d] for d in evidence if d in hs.dia_to_session_idx}
            answer_session_ids = [hs.session_ids[s] for s in sorted(gold_sessions)
                                  if s < len(hs.session_ids)]

            answer = qa.get("answer")
            answer = "" if answer is None else str(answer)
            qid = f"{sample_id}_q{qi}" + ("_abs" if is_abs else "")

            item = {
                "question_id": qid,
                "question_type": CATEGORY_TO_LME_TASK.get(cat, "single-session-user"),
                "question": qa.get("question", ""),
                "answer": answer,
                "question_date": hs.qdate,
                "haystack_sessions": hs.sessions,
                "haystack_dates": hs.dates,
                "haystack_session_ids": hs.session_ids,
                "answer_session_ids": answer_session_ids,
                "locomo_category": cat,
                "_distill_scope": sample_id,
                "_evidence": evidence,
                "_dia_to_fact_idx": hs.dia_to_fact_idx,
            }
            instances.append(LoCoMoInstance(item))

    if max_instances:
        instances = instances[:max_instances]

    logger.info(
        f"LoCoMo: {len(instances)} instances from {len(conv_ids)} conversation(s)"
        + (f", dropped categories {sorted(drop)}" if drop else "")
    )
    return LongMemEvalDataset(instances)
