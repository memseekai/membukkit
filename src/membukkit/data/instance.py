"""LongMemEval instance and dataset types for MEMBUKKIT."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from membukkit.data.base import CoreMemDataset, FactInput, QueryInput
from membukkit.time_utils import parse_datetime

logger = logging.getLogger(__name__)

ABILITY_MAP = {
    "single-session-user": "info_extraction",
    "single-session-assistant": "info_extraction",
    "single-session-preference": "info_extraction",
    "multi-session": "multi_session",
    "knowledge-update": "knowledge_update",
    "temporal-reasoning": "temporal",
}


def _parse_date(date_str: str) -> datetime:
    """Best-effort parse of LongMemEval date strings like '2022/03/21 (Mon) 14:57'."""
    return parse_datetime(date_str, default=datetime(2023, 1, 1))


def _ability_subset(question_type: str, question_id: str) -> str:
    if question_id.endswith("_abs"):
        return "abstention"
    return ABILITY_MAP.get(question_type, "info_extraction")


class LongMemEvalInstance(CoreMemDataset):
    """A single LongMemEval evaluation instance with its own haystack."""

    def __init__(self, item: Dict):
        self._item = item
        self._facts: Optional[List[FactInput]] = None
        self._queries: Optional[List[QueryInput]] = None
        self._evidence_turn_texts: Optional[List[str]] = None
        self._gold_fact_indices: Optional[List[int]] = None
        self._needle_indices: Optional[List[int]] = None

    @property
    def question_id(self) -> str:
        return self._item.get("question_id", "")

    @property
    def ability(self) -> str:
        return _ability_subset(
            self._item.get("question_type", ""),
            self.question_id,
        )

    @property
    def question_date_raw(self) -> str:
        return self._item.get("question_date", "")

    @property
    def question_date(self) -> Optional[datetime]:
        raw = self.question_date_raw
        return _parse_date(raw) if raw else None

    def _extract_facts(self) -> None:
        facts = []
        sessions = self._item.get("haystack_sessions", [])
        dates = self._item.get("haystack_dates", [])
        session_ids = self._item.get("haystack_session_ids", [])
        answer_sids = set(self._item.get("answer_session_ids", []))
        evidence_texts = []
        gold_indices = []
        needle_indices = []
        fact_session_idx: List[int] = []
        gold_session_set = set()

        for s_idx, session in enumerate(sessions):
            sid = session_ids[s_idx] if s_idx < len(session_ids) else f"s{s_idx}"
            ts = _parse_date(dates[s_idx]) if s_idx < len(dates) else datetime(2023, 1, 1)

            for t_idx, turn in enumerate(session):
                role = turn.get("role", "user")
                content = turn.get("content", "").strip()
                if not content:
                    continue

                has_answer = turn.get("has_answer", False)
                fact = FactInput(
                    text=content,
                    timestamp=ts,
                    tag="NEW_OBS",
                    source_session=str(sid),
                    source_speaker=role,
                )
                facts.append(fact)
                fact_session_idx.append(s_idx)
                fidx = len(facts) - 1

                if has_answer:
                    needle_indices.append(fidx)
                if has_answer or str(sid) in answer_sids:
                    evidence_texts.append(content)
                    gold_indices.append(fidx)
                    gold_session_set.add(s_idx)

        self._facts = facts
        self._evidence_turn_texts = evidence_texts
        self._gold_fact_indices = gold_indices
        self._needle_indices = needle_indices
        self._fact_session_idx = fact_session_idx
        self._gold_session_set = gold_session_set

    def get_facts(self) -> List[FactInput]:
        if self._facts is None:
            self._extract_facts()
        return self._facts

    def get_gold_fact_indices(self) -> List[int]:
        if self._gold_fact_indices is None:
            self._extract_facts()
        return self._gold_fact_indices or []

    def get_gold_needle_indices(self, context_window: int = 0) -> List[int]:
        if self._needle_indices is None:
            self._extract_facts()
        needles = list(self._needle_indices or [])
        if not needles:
            return self.get_gold_fact_indices()
        if context_window <= 0:
            return needles
        fact_sessions = getattr(self, "_fact_session_idx", [])
        n = len(self._facts or [])
        keep = set()
        for ni in needles:
            s = fact_sessions[ni] if ni < len(fact_sessions) else -1
            for j in range(ni - context_window, ni + context_window + 1):
                if 0 <= j < n and (j >= len(fact_sessions) or fact_sessions[j] == s):
                    keep.add(j)
        return sorted(keep)

    def get_gold_session_fact_indices(self) -> List[int]:
        if self._gold_fact_indices is None:
            self._extract_facts()
        gold_sessions = getattr(self, "_gold_session_set", set())
        fact_sessions = getattr(self, "_fact_session_idx", [])
        return [i for i, s in enumerate(fact_sessions) if s in gold_sessions]

    def get_queries(self) -> List[QueryInput]:
        if self._queries is None:
            ability = self.ability
            answer = self._item.get("answer", "")
            if answer is None:
                answer = ""
            self._queries = [QueryInput(
                text=self._item.get("question", ""),
                query_type=ability,
                ground_truth=str(answer),
                evidence=self._item.get("answer_session_ids", []),
                category=None,
            )]
        return self._queries

    def get_utility_matrix(self) -> np.ndarray:
        from membukkit.eval.metrics import f1_score as _f1
        facts = self.get_facts()
        queries = self.get_queries()
        S = np.zeros((len(facts), len(queries)), dtype=np.float32)
        for q_idx, q in enumerate(queries):
            gt = q.ground_truth or ""
            if not gt:
                continue
            for i, f in enumerate(facts):
                S[i, q_idx] = _f1(f.text, gt)
        return S

    def evaluate(self, predictions: Any) -> Dict[str, float]:
        from membukkit.eval.metrics import f1_score as _f1
        queries = self.get_queries()
        scores = []
        for q_idx, q in enumerate(queries):
            pred = predictions.get(q_idx, "")
            gt = q.ground_truth or ""
            scores.append(_f1(pred, gt))
        return {"f1": float(np.mean(scores)) if scores else 0.0}


class LongMemEvalDataset:
    """Full LongMemEval benchmark: 500 instances grouped by ability subset."""

    def __init__(self, instances: List[LongMemEvalInstance]):
        self.instances = instances
        self._by_ability: Optional[Dict[str, List[LongMemEvalInstance]]] = None

    @property
    def by_ability(self) -> Dict[str, List[LongMemEvalInstance]]:
        if self._by_ability is None:
            groups: Dict[str, List[LongMemEvalInstance]] = {}
            for inst in self.instances:
                groups.setdefault(inst.ability, []).append(inst)
            self._by_ability = groups
        return self._by_ability

    def get_ability_names(self) -> List[str]:
        return sorted(self.by_ability.keys())

    def filter_by_ability(self, ability: str) -> List[LongMemEvalInstance]:
        return self.by_ability.get(ability, [])

    def summary(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self.by_ability.items()}
