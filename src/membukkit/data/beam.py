"""BEAM benchmark loader for MEMBUKKIT.

BEAM (Tavakoli et al., ICLR 2026, arXiv:2510.27246) evaluates long-term
conversational memory across 4 scales (100K / 500K / 1M / 10M tokens) and
10 memory abilities, with rubric ("nugget") based LLM-judge scoring.

Data layout (official repo, github.com/mohammadtavakoli78/BEAM):
  chats/<scale>/<conv_id>/chat.json
      -> list of "batches"; each batch has one date time_anchor and a list
         of [user, assistant] turn pairs. We treat each batch as a session.
  chats/<scale>/<conv_id>/probing_questions/probing_questions.json
      -> dict of 10 ability categories, 2 questions each (20 per conv).
         Every question has `question` + `rubric` (list of nugget strings);
         the gold-answer field name varies by category.

Conversations per scale: 100K: 20, 500K: 35, 1M: 35, 10M: 10.

User turn contents end with generator artifacts like " ->-> 2,3" (question
index markers); these are stripped on ingest. Time anchors look like
"March-15-2024" and are attached to every fact from that batch.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from membukkit.data.base import CoreMemDataset, FactInput, QueryInput

logger = logging.getLogger(__name__)

RAW_BASE = "https://raw.githubusercontent.com/mohammadtavakoli78/BEAM/main"

SCALES: Dict[str, int] = {"100K": 20, "500K": 35, "1M": 35, "10M": 10}

CATEGORIES = [
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
]

# Gold-answer field varies by category; first present field wins.
_ANSWER_FIELDS = (
    "answer",
    "ideal_response",
    "ideal_answer",
    "ideal_summary",
    "expected_compliance",
)

# Observed forms: "->-> 2,3", "->-> 2,N/A", "->-> 5,14)", "->-> 2,22, 24"
_MARKER_RE = re.compile(r"\s*->->\s*\d+\s*,\s*(?:\d+|N/A)(?:\s*,\s*\d+)*\)?\s*$")


def _strip_marker(text: str) -> str:
    return _MARKER_RE.sub("", text).strip()


def _parse_anchor(anchor: Optional[str]) -> datetime:
    """Parse BEAM time anchors like 'March-15-2024'."""
    if anchor:
        for fmt in ("%B-%d-%Y", "%b-%d-%Y", "%B %d, %Y"):
            try:
                return datetime.strptime(anchor.strip(), fmt)
            except ValueError:
                continue
    return datetime(2024, 1, 1)


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.rename(dest)
    return dest


@dataclass
class BeamQuestion:
    """A single BEAM probing question with its official judging payload."""

    category: str
    index: int  # 0 or 1 within category
    question: str
    gold_answer: str
    rubric: List[str]
    difficulty: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def qid(self) -> str:
        return f"{self.category}/{self.index}"


def _flatten_chat(chat: List[Dict]) -> List[Dict]:
    """Normalize chat.json across scales.

    100K/500K/1M: a flat list of batches ({"batch_number","turns",...}).
    10M: a list of {"plan-N": [batches]} wrappers; flatten in plan order.
    """
    if chat and isinstance(chat[0], dict) and "turns" not in chat[0]:
        flat: List[Dict] = []
        for wrapper in chat:
            for batches in wrapper.values():
                flat.extend(batches)
        return flat
    return chat


class BeamInstance(CoreMemDataset):
    """One BEAM conversation: its own haystack + 20 probing questions."""

    def __init__(self, scale: str, conv_id: int, chat: List[Dict], pq: Dict[str, List[Dict]]):
        self.scale = scale
        self.conv_id = conv_id
        self._chat = _flatten_chat(chat)
        self._pq = pq
        self._facts: Optional[List[FactInput]] = None
        self._questions: Optional[List[BeamQuestion]] = None

    @property
    def instance_id(self) -> str:
        return f"beam_{self.scale}_{self.conv_id}"

    def _extract_facts(self) -> None:
        facts: List[FactInput] = []
        for b_idx, batch in enumerate(self._chat):
            batch_no = batch.get("batch_number", b_idx + 1)
            sid = f"{self.instance_id}_b{batch_no}"
            # One time anchor per batch, carried on user turns.
            anchor = None
            for pair in batch.get("turns", []):
                for turn in pair:
                    if turn.get("time_anchor"):
                        anchor = turn["time_anchor"]
                        break
                if anchor:
                    break
            ts = _parse_anchor(anchor)

            for pair in batch.get("turns", []):
                for turn in pair:
                    role = turn.get("role", "user")
                    content = _strip_marker(turn.get("content", "") or "")
                    if not content:
                        continue
                    facts.append(
                        FactInput(
                            text=content,
                            timestamp=ts,
                            tag="NEW_OBS",
                            source_session=sid,
                            source_speaker=role,
                        )
                    )
        self._facts = facts

    def get_facts(self) -> List[FactInput]:
        if self._facts is None:
            self._extract_facts()
        return self._facts

    @property
    def questions(self) -> List[BeamQuestion]:
        if self._questions is None:
            out: List[BeamQuestion] = []
            for cat in CATEGORIES:
                for i, item in enumerate(self._pq.get(cat, [])):
                    gold = ""
                    for f in _ANSWER_FIELDS:
                        if item.get(f):
                            gold = str(item[f])
                            break
                    out.append(
                        BeamQuestion(
                            category=cat,
                            index=i,
                            question=item.get("question", ""),
                            gold_answer=gold,
                            rubric=list(item.get("rubric", [])),
                            difficulty=item.get("difficulty", ""),
                            raw=item,
                        )
                    )
            self._questions = out
        return self._questions

    def get_queries(self) -> List[QueryInput]:
        return [
            QueryInput(
                text=q.question,
                query_type=q.category,
                ground_truth=q.gold_answer,
                evidence=None,
                category=None,
            )
            for q in self.questions
        ]

    def get_utility_matrix(self) -> np.ndarray:
        raise NotImplementedError("BEAM has no per-fact utility labels")

    def evaluate(self, predictions: Any) -> Dict[str, float]:
        raise NotImplementedError(
            "BEAM must be scored with membukkit.eval.beam_official (LLM judge)"
        )


class BeamDataset:
    """One BEAM scale split (e.g. all 20 conversations at 100K)."""

    def __init__(self, scale: str, instances: List[BeamInstance]):
        self.scale = scale
        self.instances = instances

    def summary(self) -> Dict[str, int]:
        return {
            "conversations": len(self.instances),
            "questions": sum(len(i.questions) for i in self.instances),
        }


class BeamQAInstance(CoreMemDataset):
    """One BEAM probing question, shaped like a LongMemEval instance.

    All 20 questions of a conversation share the same haystack lists (by
    reference) and the same `distill_scope_id`, so the eval CLI ingests each
    conversation exactly once (the LoCoMo pattern). Batches are chunked into
    windows of `pairs_per_session` turn pairs (BEAM batches run ~30-40K tokens,
    far past what one distill call should see); every chunk keeps its batch's
    date anchor.
    """

    def __init__(
        self,
        question_id: str,
        scope_id: str,
        item: Dict[str, Any],
        question: BeamQuestion,
    ):
        self.question_id = question_id
        self.distill_scope_id = scope_id
        self._item = item
        self.beam_question = question
        self._queries: Optional[List[QueryInput]] = None

    @property
    def ability(self) -> str:
        return self.beam_question.category

    @property
    def beam_rubric(self) -> List[str]:
        return self.beam_question.rubric

    @property
    def question_date_raw(self) -> str:
        return self._item.get("question_date", "")

    def get_facts(self) -> List[FactInput]:
        facts: List[FactInput] = []
        sessions = self._item.get("haystack_sessions", [])
        dates = self._item.get("haystack_dates", [])
        for s_idx, session in enumerate(sessions):
            ts = _parse_anchor(dates[s_idx] if s_idx < len(dates) else None)
            for turn in session:
                content = (turn.get("content", "") or "").strip()
                if content:
                    facts.append(
                        FactInput(
                            text=content,
                            timestamp=ts,
                            tag="NEW_OBS",
                            source_session=f"s{s_idx}",
                            source_speaker=turn.get("role", "user"),
                        )
                    )
        return facts

    def get_queries(self) -> List[QueryInput]:
        if self._queries is None:
            q = self.beam_question
            self._queries = [
                QueryInput(
                    text=q.question,
                    query_type=q.category,
                    ground_truth=q.gold_answer,
                    evidence=None,
                    category=None,
                )
            ]
        return self._queries

    def get_utility_matrix(self) -> np.ndarray:
        raise NotImplementedError("BEAM has no per-fact utility labels")

    def evaluate(self, predictions: Any) -> Dict[str, float]:
        raise NotImplementedError(
            "BEAM must be scored with membukkit.eval.beam_official (LLM judge)"
        )


class BeamQADataset:
    """Flat list of per-question instances, CLI-compatible (.instances)."""

    def __init__(self, scale: str, instances: List[BeamQAInstance]):
        self.scale = scale
        self.instances = instances

    def summary(self) -> Dict[str, int]:
        return {
            "questions": len(self.instances),
            "conversations": len({i.distill_scope_id for i in self.instances}),
        }


def _chunk_conversation(
    chat: List[Dict], pairs_per_session: int
) -> Tuple[List[List[Dict[str, str]]], List[str]]:
    """Chunk batches into sessions of at most `pairs_per_session` turn pairs.

    Returns (sessions, dates): sessions in the {"role","content"} format the
    eval CLI / MemorySystem.ingest expect, dates as ISO strings (one per
    session, the enclosing batch's time anchor).
    """
    sessions: List[List[Dict[str, str]]] = []
    dates: List[str] = []
    for b_idx, batch in enumerate(chat):
        anchor = None
        for pair in batch.get("turns", []):
            for turn in pair:
                if turn.get("time_anchor"):
                    anchor = turn["time_anchor"]
                    break
            if anchor:
                break
        iso = _parse_anchor(anchor).date().isoformat()

        pairs = batch.get("turns", [])
        for start in range(0, len(pairs), pairs_per_session):
            chunk: List[Dict[str, str]] = []
            for pair in pairs[start : start + pairs_per_session]:
                for turn in pair:
                    content = _strip_marker(turn.get("content", "") or "")
                    if content:
                        chunk.append({"role": turn.get("role", "user"), "content": content})
            if chunk:
                sessions.append(chunk)
                dates.append(iso)
    return sessions, dates


def load_beam_qa(
    scale: str = "100K",
    cache_dir: Optional[str] = None,
    max_conversations: Optional[int] = None,
    pairs_per_session: int = 6,
) -> BeamQADataset:
    """Load BEAM as per-question instances for the eval CLI.

    The question date is the conversation's last time anchor (probing
    questions are asked after the full history).
    """
    ds = load_beam(scale, cache_dir=cache_dir, max_instances=max_conversations)
    instances: List[BeamQAInstance] = []
    for conv in ds.instances:
        sessions, dates = _chunk_conversation(conv._chat, pairs_per_session)
        if not sessions:
            continue
        qdate = max(dates)
        scope = conv.instance_id  # beam_{scale}_{conv_id}
        for q in conv.questions:
            item = {
                "haystack_sessions": sessions,  # shared by reference
                "haystack_dates": dates,
                "question_type": q.category,
                "question": q.question,
                "answer": q.gold_answer,
                "question_date": qdate,
                "beam_conv": scope,
            }
            instances.append(
                BeamQAInstance(
                    question_id=f"{scope}_{q.category}_{q.index}",
                    scope_id=scope,
                    item=item,
                    question=q,
                )
            )
    out = BeamQADataset(scale, instances)
    logger.info(f"BEAM-QA {scale} loaded: {out.summary()}")
    return out


def load_beam(
    scale: str = "100K",
    cache_dir: Optional[str] = None,
    max_instances: Optional[int] = None,
) -> BeamDataset:
    """Download (if needed) and load one BEAM scale split.

    Args:
        scale: one of "100K", "500K", "1M", "10M"
        cache_dir: where to cache raw files (default ~/.cache/membukkit/beam)
        max_instances: cap number of conversations (dev/debug)
    """
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {list(SCALES)}, got {scale!r}")

    root = Path(cache_dir or Path.home() / ".cache" / "membukkit" / "beam") / scale
    n = SCALES[scale]
    if max_instances is not None:
        n = min(n, max_instances)

    instances: List[BeamInstance] = []
    for conv_id in range(1, n + 1):
        chat_path = _download(
            f"{RAW_BASE}/chats/{scale}/{conv_id}/chat.json",
            root / str(conv_id) / "chat.json",
        )
        pq_path = _download(
            f"{RAW_BASE}/chats/{scale}/{conv_id}/probing_questions/probing_questions.json",
            root / str(conv_id) / "probing_questions.json",
        )
        with open(chat_path) as f:
            chat = json.load(f)
        with open(pq_path) as f:
            pq = json.load(f)
        instances.append(BeamInstance(scale, conv_id, chat, pq))

    ds = BeamDataset(scale, instances)
    logger.info(f"BEAM {scale} loaded: {ds.summary()}")
    return ds
