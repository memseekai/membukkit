"""
Query-type router for the structural memory index.

The core finding of the disambiguation experiments: organization, not retrieval,
is the bottleneck for memory reasoning — and the *right* organization is
query-type-dependent (temporal ordering helps temporal/knowledge-update queries
but is neutral-to-harmful elsewhere). A fixed "always sort by date" strategy is
therefore a trick, not a method.

This router detects what each query needs and routes it to the organization that
helps, emitting an interpretable rationale (the routing trace IS the explanation).

It is intentionally rule-based and transparent first; `Organization` and
`QueryClass` form a stable interface so a learned classifier can drop in later
without changing downstream code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class QueryClass(str, Enum):
    TEMPORAL = "temporal"
    KNOWLEDGE_UPDATE = "knowledge_update"
    AGGREGATION = "aggregation"
    GENERAL = "general"


class Organization(str, Enum):
    """How retrieved evidence should be organized before reasoning."""

    TEMPORAL = "temporal"
    PLAIN = "plain"


_TEMPORAL_CUES = [
    "when",
    "what time",
    "what date",
    "how long",
    "how long ago",
    "ago",
    "before",
    "after",
    "first",
    "earliest",
    "last time",
    "latest",
    "most recent",
    "recently",
    "prior to",
    "since",
    "until",
    "which came first",
    "in what order",
    "in order",
    "order from",
    "first to last",
    "came first",
    "happened first",
    "chronolog",
    "timeline",
    "duration",
    "between",
]
_KNOWLEDGE_UPDATE_CUES = [
    "now",
    "currently",
    "current",
    "still",
    "anymore",
    "no longer",
    "these days",
    "nowadays",
    "changed",
    "change",
    "switch",
    "switched",
    "updated",
    "update",
    "moved",
    "new ",
    "latest",
    "as of",
]
_AGGREGATION_CUES = [
    "how many",
    "how much",
    "count",
    "number of",
    "list all",
    "list the",
    "all the",
    "total",
    "in total",
    "every",
    "each of",
]

_DATE_RE = re.compile(
    r"\b(19|20)\d{2}\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"(uary|ruary|ch|il|e|y|ust|tember|ober|ember)?\b",
    re.IGNORECASE,
)

_RECOMMENDATION_RE = re.compile(
    r"\b(recommend\w*|suggest\w*|propose)\b"
    r"|what should i|what would you (?:recommend|suggest|do)"
    r"|any (?:advice|ideas?|tips|suggestions?|recommendations?|thoughts)"
    r"|do you have any (?:advice|ideas?|tips|suggestions?|recommendations?)"
    r"|(?:do you think|what do you think).{0,40}(?:good idea|should|better|worth|might|would)"
    r"|good idea to|help me (?:find|pick|choose|decide)|worth (?:it|trying|buying)",
    re.IGNORECASE,
)


def is_recommendation_query(query_text: str) -> bool:
    """True if the query is an open-ended recommendation/advice request.

    Such queries are non-extractive: the answer must be tailored to the user's
    preferences rather than retrieved verbatim, so they route to a recommendation
    reader instead of the factual reader.
    """
    return bool(_RECOMMENDATION_RE.search(query_text or ""))


_CHANGELOG_RE = re.compile(
    r"\bwhat(?:'s|s| has| have)?\s+chang"
    r"|\bwhat\s+changed\b"
    r"|\bchang(?:ed|es|ing)\s+(?:recently|lately|over\s+time)\b"
    r"|\brecently\s+chang"
    r"|\bwhat(?:'s|s)?\s+new\b"
    r"|\bwhat\s+(?:was|were|got|have\s+been|has\s+been)\s+updat"
    r"|\bupdat(?:ed|es|ing)\s+(?:recently|lately)\b"
    r"|\bwhat\s+has\s+changed\b"
    r"|\bover\s+time\b",
    re.IGNORECASE,
)


def is_changelog_query(query_text: str) -> bool:
    """True when the user wants a memory-timeline changelog, not wall-clock recency.

    These questions should summarize the latest dated state changes in evidence
    on or before as-of — not abstain because memories are older than 'today'.
    """
    return bool(_CHANGELOG_RE.search(query_text or ""))


_READER_ROUTER: Optional["QueryRouter"] = None


def is_reasoning_query(query_text: str) -> bool:
    """True if the query needs multi-fact SYNTHESIS — temporal arithmetic,
    cross-session aggregation/counting, or resolving a changed (knowledge-update)
    value — and should route to a step-by-step reasoning reader (E21 m1).

    Text-only via the rule-based router (no ability_hint) -> leakage-free. The
    oracle probe showed a reasoning reader lifts the gold-evidence ceiling on
    multi-session (0.68->0.81) and temporal (0.78->0.87); these are the classes
    that benefit. GENERAL single-fact lookups stay on the plain dated reader.
    """
    global _READER_ROUTER
    if _READER_ROUTER is None:
        _READER_ROUTER = QueryRouter()
    qc = _READER_ROUTER.route(query_text).query_class
    return qc in (QueryClass.TEMPORAL, QueryClass.AGGREGATION, QueryClass.KNOWLEDGE_UPDATE)


def has_aggregation_cues(query_text: str) -> bool:
    """True when the query contains counting/totaling language (how many, total,
    list all, ...), regardless of the router's final class.

    Distinct from ``QueryClass.AGGREGATION``: router precedence sends how-many
    questions that also mention a date or ordering word (\"how many runs did I
    miss in March\") to TEMPORAL, but they still need exhaustive enumeration —
    one missed mention breaks the count. Used by the eval's aggregation-routing
    experiment to widen retrieval for exactly these queries.
    """
    return bool(_find_cues((query_text or "").lower(), _AGGREGATION_CUES))


_ASSISTANT_RECALL_RE = re.compile(
    r"\byou (provided|gave|created|generated|suggested|recommended|told|said"
    r"|wrote|listed|shared|mentioned|helped)"
    r"|our (previous |last |earlier )?(conversation|discussion|chat)"
    r"|we (discussed|talked)",
    re.IGNORECASE,
)


def has_assistant_recall_cues(query_text: str) -> bool:
    """True when the query asks to recall something the ASSISTANT said or made
    ("you provided a list...", "our previous conversation about...").

    These are needle queries over assistant turns: the answer is one specific
    prior response, usually long and verbatim-lane, so it ranks deeper than
    user-fact needles and benefits from a wider retrieval depth. The cue is
    query-surface only (no dataset labels involved).
    """
    return bool(_ASSISTANT_RECALL_RE.search(query_text or ""))


_BROAD_CONTEXT_RE = re.compile(
    r"\b(summar\w*|recap|overview|comprehensive|walk me through"
    r"|how (?:has|have) .{0,60}?(?:progress|evolv|chang|develop)"
    r"|(?:progression|evolution|journey) of)\b",
    re.IGNORECASE | re.DOTALL,
)


def has_broad_context_cues(query_text: str) -> bool:
    """True when the query asks for a synthesis over the whole history
    ("summarize...", "comprehensive overview", "how has X evolved").

    These are coverage-bound: the answer is graded on how many distinct
    aspects it covers (e.g. BEAM summarization rubrics), so a standard top-k
    starves the reader. Query-surface only, no dataset labels involved.
    """
    return bool(_BROAD_CONTEXT_RE.search(query_text or ""))


def is_temporal_query(query_text: str) -> bool:
    """True only for TEMPORAL-class queries (explicit time intent / dates / ordering).

    Temporal answers need the WHOLE timeline — a single missed dated fact breaks the
    chain — so these queries are recall-bound and benefit from a deeper bucket scan,
    unlike aggregation/knowledge-update whose recall is already near-ceiling at 0.3.
    """
    global _READER_ROUTER
    if _READER_ROUTER is None:
        _READER_ROUTER = QueryRouter()
    return _READER_ROUTER.route(query_text).query_class == QueryClass.TEMPORAL


@dataclass
class RoutingDecision:
    """A routed query: its class, the chosen organization, and why (for XAI)."""

    query_class: QueryClass
    organization: Organization
    rationale: str
    matched_cues: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def trace(self) -> str:
        cue_str = f" [cues: {', '.join(self.matched_cues)}]" if self.matched_cues else ""
        return (
            f"{self.query_class.value} -> organize:{self.organization.value} "
            f"({self.rationale}){cue_str}"
        )


def _find_cues(text: str, cues: List[str]) -> List[str]:
    return [c for c in cues if c in text]


class QueryRouter:
    """Routes a query to the memory organization that helps it.

    Rule-based and interpretable. Precedence reflects the empirical findings:
    temporal/knowledge-update queries benefit from chronological organization;
    aggregation and general queries do not (and temporal ordering can hurt them),
    so they stay plain.
    """

    def route(
        self,
        query_text: str,
        query_frame: Optional[object] = None,
        ability_hint: Optional[str] = None,
    ) -> RoutingDecision:
        """Route a query to an organization strategy.

        Args:
            query_text: natural language question
            query_frame: optional parsed QueryFrame from the LLM parser
            ability_hint: optional dataset ability label (e.g. LongMemEval's
                ``temporal``, ``knowledge_update``). Used when the question
                text alone does not carry temporal/update cues — common on LME.
        """
        q = (query_text or "").lower()

        if ability_hint:
            ah = ability_hint.lower().replace("-", "_")
            if ah in ("temporal", "temporal_reasoning"):
                return RoutingDecision(
                    QueryClass.TEMPORAL,
                    Organization.TEMPORAL,
                    f"dataset ability={ability_hint}: temporal reasoning needs chronological evidence",
                    matched_cues=[f"ability:{ability_hint}"],
                )
            if ah in ("knowledge_update", "knowledge-update"):
                return RoutingDecision(
                    QueryClass.KNOWLEDGE_UPDATE,
                    Organization.TEMPORAL,
                    f"dataset ability={ability_hint}: conflicting facts need recency ordering",
                    matched_cues=[f"ability:{ability_hint}"],
                )
            if ah in ("multi_session", "multi-session"):
                return RoutingDecision(
                    QueryClass.AGGREGATION,
                    Organization.TEMPORAL,
                    f"dataset ability={ability_hint}: cross-session synthesis benefits chronology",
                    matched_cues=[f"ability:{ability_hint}"],
                )
            if ah in ("info_extraction", "abstention"):
                return RoutingDecision(
                    QueryClass.GENERAL,
                    Organization.PLAIN,
                    f"dataset ability={ability_hint}: direct lookup, plain order suffices",
                    matched_cues=[f"ability:{ability_hint}"],
                )

        frame_time_op = getattr(query_frame, "time_op", None) if query_frame else None
        frame_requires_agg = (
            getattr(query_frame, "requires_aggregation", False) if query_frame else False
        )

        temporal_cues = _find_cues(q, _TEMPORAL_CUES)
        ku_cues = _find_cues(q, _KNOWLEDGE_UPDATE_CUES)
        agg_cues = _find_cues(q, _AGGREGATION_CUES)
        has_date = bool(_DATE_RE.search(q))

        if ku_cues:
            return RoutingDecision(
                QueryClass.KNOWLEDGE_UPDATE,
                Organization.TEMPORAL,
                "change/recency language implies the current value depends on time order",
                matched_cues=ku_cues,
            )

        if temporal_cues or has_date or frame_time_op:
            cues = list(temporal_cues)
            if has_date:
                cues.append("<date>")
            if frame_time_op:
                cues.append(f"frame.time_op={frame_time_op}")
            return RoutingDecision(
                QueryClass.TEMPORAL,
                Organization.TEMPORAL,
                "explicit temporal intent: chronological ordering aids reasoning",
                matched_cues=cues,
            )

        if agg_cues or frame_requires_agg:
            cues = list(agg_cues)
            if frame_requires_agg:
                cues.append("frame.requires_aggregation")
            return RoutingDecision(
                QueryClass.AGGREGATION,
                Organization.PLAIN,
                "aggregation query: temporal ordering does not help; keep plain",
                matched_cues=cues,
            )

        return RoutingDecision(
            QueryClass.GENERAL,
            Organization.PLAIN,
            "no temporal/update/aggregation signal: plain retrieval order",
        )
