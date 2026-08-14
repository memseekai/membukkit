"""Translate a query into Turbopuffer filter clauses (time + entity axes).

The in-RAM ``multiaxis`` path unioned entity/time matches by rebuilding indices
per query. Against a DB those axes are just filter clauses, so the per-query
KMeans rebuild disappears. This module lifts the year/month parsing and entity
extraction onto the query side and emits Turbopuffer-shaped filter tuples.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional, Tuple

from membukkit.retrieval.bucket_index import extract_entities
from membukkit.time_utils import TS_UNKNOWN, day_range

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_MONTH_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"(uary|ruary|ch|il|e|y|ust|tember|ober|ember)?\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?"
    r"(?:Z|[+-]\d{2}:\d{2})?)?\b"
)
_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def query_entities(query_text: str) -> List[str]:
    """Canonical entities mentioned in the query (for a ``ContainsAny`` filter)."""
    return sorted(extract_entities(query_text or ""))


def query_time_range(query_text: str) -> Optional[Tuple[datetime, datetime]]:
    """Infer an inclusive [start, end) datetime window from explicit dates in the query.

    Returns None when the query carries no explicit year/month, so callers fall
    back to no temporal filter.
    """
    text = query_text or ""
    iso_match = _ISO_DATE_RE.search(text)
    if iso_match:
        rng = day_range(iso_match.group(0))
        if rng is not None:
            return rng

    years = [int(m.group(0)) for m in _YEAR_RE.finditer(text)]
    months = [_MONTHS[m.group(1).lower()[:3]] for m in _MONTH_RE.finditer(text)]
    if years and months:
        y, mo = min(years), min(months)
        start = datetime(y, mo, 1)
        end = datetime(y + (mo == 12), (mo % 12) + 1, 1)
        return (start, end)
    if years:
        y = min(years)
        return (datetime(y, 1, 1), datetime(y + 1, 1, 1))
    return None


def build_filter(
    *,
    topic_buckets: Optional[List[int]] = None,
    entities: Optional[List[str]] = None,
    time_range: Optional[Tuple[datetime, datetime]] = None,
    live_only: bool = True,
    include_undated: bool = True,
    kind: Optional[str] = None,
) -> Optional[list]:
    """Compose a Turbopuffer ``And(...)`` filter from the active axes.

    In gated mode the topic-bucket clause and the entity/time clauses are OR'd so
    entity/time "needles" outside the opened topic buckets are still retrievable
    (the intent of the old multiaxis union); the live-only clause is AND'd on top.

    ``include_undated`` keeps facts stored at the ``TS_UNKNOWN`` sentinel visible
    to date-ranged queries — an undated fact can't be ruled out by date.

    ``kind`` (verbatim/atomic) is AND'd on top so a union lane sees only its own
    facts. It is intentionally a hard constraint, not part of the OR'd axis.
    """
    clauses: list = []
    if live_only:
        clauses.append(["superseded_by", "Eq", ""])
    if kind:
        clauses.append(["kind", "Eq", kind])

    axis: list = []
    if topic_buckets:
        axis.append(["topic_bucket", "In", list(topic_buckets)])
    if entities:
        axis.append(["entities", "ContainsAny", list(entities)])
    if time_range is not None:
        in_range = ["And", [["ts", "Gte", time_range[0]], ["ts", "Lt", time_range[1]]]]
        if include_undated:
            axis.append(["Or", [in_range, ["ts", "Eq", TS_UNKNOWN]]])
        else:
            axis.append(in_range)

    if len(axis) == 1:
        clauses.append(axis[0])
    elif len(axis) > 1:
        clauses.append(["Or", axis])

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return ["And", clauses]
