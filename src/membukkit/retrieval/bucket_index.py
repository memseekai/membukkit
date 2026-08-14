"""
Inspectable structural bucket index for memory.

Facts are assigned to buckets along interpretable axes; a query activates a
subset of buckets; the activation trace IS the explanation. Everything is
serializable so posthoc interpretability analysis can answer:
  - which bucket(s) did query Q activate, and why?
  - which facts live in bucket B?
  - which bucket did the answer's evidence come from?

Bucket axes:
  - entity  (symbolic): canonical entity / relation -> fact ids
  - time    (symbolic): coarse year-month window    -> fact ids
  - topic   (learned) : KMeans cluster over embeddings -> fact ids
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_CAP_RE = re.compile(r"\b[A-Z][a-zA-Z]+\b")
_STOP_CAPS = {
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "the",
    "a",
    "an",
    "is",
    "are",
    "do",
    "does",
    "did",
    "i",
    "my",
    "you",
    "your",
    "it",
    "he",
    "she",
}
_REL_ALIASES = {
    "mom": "mother",
    "mum": "mother",
    "mama": "mother",
    "mommy": "mother",
    "dad": "father",
    "papa": "father",
    "daddy": "father",
    "sis": "sister",
    "bro": "brother",
    "grandma": "grandmother",
    "granny": "grandmother",
    "grandpa": "grandfather",
    "hubby": "husband",
}
_REL_TERMS = {
    "mother",
    "father",
    "sister",
    "brother",
    "son",
    "daughter",
    "spouse",
    "husband",
    "wife",
    "grandmother",
    "grandfather",
    "parent",
    "sibling",
    "child",
    "colleague",
    "manager",
    "friend",
    "boss",
    "teacher",
    "doctor",
    "neighbor",
    "cousin",
    "uncle",
    "aunt",
    "nephew",
    "niece",
}


def extract_entities(text: str) -> Set[str]:
    """Proper-noun surface forms + alias-normalized relationship terms."""
    ents = {t.lower() for t in _CAP_RE.findall(text)}
    ents = {e for e in ents if e not in _STOP_CAPS}
    for t in (tok.lower() for tok in _TOKEN_RE.findall(text)):
        canon = _REL_ALIASES.get(t, t)
        if canon in _REL_TERMS:
            ents.add(canon)
    return ents


def time_bucket_key(t: Optional[datetime]) -> str:
    if t is None:
        return "unknown"
    return f"{t.year:04d}-{t.month:02d}"


@dataclass
class BucketActivation:
    """Which buckets a query activated and why — the routing explanation."""

    entity_buckets: List[str] = field(default_factory=list)
    time_buckets: List[str] = field(default_factory=list)
    topic_buckets: List[int] = field(default_factory=list)
    rationale: List[str] = field(default_factory=list)

    def activated_fact_ids(self, index: "StructuralBucketIndex") -> Set[int]:
        ids: Set[int] = set()
        for e in self.entity_buckets:
            ids |= set(index.entity_to_facts.get(e, []))
        for tb in self.time_buckets:
            ids |= set(index.time_to_facts.get(tb, []))
        for tc in self.topic_buckets:
            ids |= set(index.topic_to_facts.get(tc, []))
        return ids

    def to_dict(self) -> Dict:
        return {
            "entity_buckets": self.entity_buckets,
            "time_buckets": self.time_buckets,
            "topic_buckets": self.topic_buckets,
            "rationale": self.rationale,
        }


class StructuralBucketIndex:
    """Assigns facts to inspectable buckets and routes queries to bucket subsets.

    Indexed over a single memory bank (one character / instance / user).
    """

    def __init__(self, n_topics: int = 12, random_state: int = 0):
        self.n_topics = n_topics
        self.random_state = random_state

        self.fact_texts: List[str] = []
        self.fact_times: List[Optional[datetime]] = []
        self.fact_entities: List[Set[str]] = []
        self.fact_topic: List[int] = []

        self.entity_to_facts: Dict[str, List[int]] = defaultdict(list)
        self.time_to_facts: Dict[str, List[int]] = defaultdict(list)
        self.topic_to_facts: Dict[int, List[int]] = defaultdict(list)
        self.topic_centroids: Optional[np.ndarray] = None

    def build(
        self,
        fact_texts: Sequence[str],
        fact_times: Sequence[Optional[datetime]],
        fact_embs: Optional[np.ndarray] = None,
    ) -> "StructuralBucketIndex":
        self.fact_texts = list(fact_texts)
        self.fact_times = list(fact_times)
        n = len(self.fact_texts)

        self.fact_entities = [extract_entities(t) for t in self.fact_texts]
        for i, ents in enumerate(self.fact_entities):
            for e in ents:
                self.entity_to_facts[e].append(i)
        for i, t in enumerate(self.fact_times):
            self.time_to_facts[time_bucket_key(t)].append(i)

        self.fact_topic = [-1] * n
        if fact_embs is not None and n >= 2:
            k = min(self.n_topics, n)
            try:
                from sklearn.cluster import KMeans

                km = KMeans(n_clusters=k, random_state=self.random_state, n_init=4)
                labels = km.fit_predict(np.asarray(fact_embs))
                self.topic_centroids = km.cluster_centers_
                for i, lbl in enumerate(labels):
                    self.fact_topic[i] = int(lbl)
                    self.topic_to_facts[int(lbl)].append(i)
            except Exception:
                pass
        return self

    def route(
        self,
        query_text: str,
        query_emb: Optional[np.ndarray] = None,
        topic_top: int = 2,
        has_time_intent: bool = False,
    ) -> BucketActivation:
        """Activate buckets for a query and record why (interpretable trace)."""
        act = BucketActivation()

        q_ents = extract_entities(query_text)
        for e in q_ents:
            if e in self.entity_to_facts:
                act.entity_buckets.append(e)
        if act.entity_buckets:
            act.rationale.append(f"entity match: {', '.join(act.entity_buckets)}")

        if has_time_intent and self.time_to_facts:
            act.time_buckets = [k for k in self.time_to_facts if k != "unknown"]
            act.rationale.append("temporal intent -> all dated time buckets")

        if query_emb is not None and self.topic_centroids is not None:
            q = np.asarray(query_emb, dtype=np.float32)
            qn = q / (np.linalg.norm(q) + 1e-8)
            cn = self.topic_centroids / (
                np.linalg.norm(self.topic_centroids, axis=1, keepdims=True) + 1e-8
            )
            sims = cn @ qn
            top = np.argsort(sims)[::-1][:topic_top]
            act.topic_buckets = [int(t) for t in top]
            act.rationale.append(
                f"nearest topic clusters: {act.topic_buckets} "
                f"(sim {', '.join(f'{sims[t]:.2f}' for t in top)})"
            )
        return act

    def bucket_summary(self) -> Dict:
        return {
            "n_facts": len(self.fact_texts),
            "n_entity_buckets": len(self.entity_to_facts),
            "n_time_buckets": len(self.time_to_facts),
            "n_topic_buckets": len([t for t in self.topic_to_facts]),
            "entity_bucket_sizes": {
                e: len(v)
                for e, v in sorted(self.entity_to_facts.items(), key=lambda x: -len(x[1]))[:20]
            },
            "time_bucket_sizes": {k: len(v) for k, v in sorted(self.time_to_facts.items())},
            "topic_bucket_sizes": {str(k): len(v) for k, v in sorted(self.topic_to_facts.items())},
        }

    def topic_exemplars(self, max_per_topic: int = 3) -> Dict[int, List[str]]:
        """Representative fact texts per learned topic cluster (for labeling)."""
        out: Dict[int, List[str]] = {}
        for tc, ids in self.topic_to_facts.items():
            out[tc] = [self.fact_texts[i] for i in ids[:max_per_topic]]
        return out

    def to_dict(self) -> Dict:
        """Full serializable snapshot for posthoc interpretability analysis."""
        return {
            "summary": self.bucket_summary(),
            "fact_assignments": [
                {
                    "fact_idx": i,
                    "text": self.fact_texts[i][:200],
                    "entities": sorted(self.fact_entities[i]),
                    "time_bucket": time_bucket_key(self.fact_times[i]),
                    "topic_bucket": self.fact_topic[i],
                }
                for i in range(len(self.fact_texts))
            ],
            "topic_exemplars": {str(k): v for k, v in self.topic_exemplars().items()},
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
