"""Date/time helpers for MEMBUKKIT public boundaries and storage.

ISO8601 is the canonical string format. Legacy slash-date formats remain
accepted for existing benchmarks and callers.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import TypeAlias

DateLike: TypeAlias = datetime | date | str | None

# Storage sentinel for "fact has no known date". Datetime columns can't hold
# NULL, so undated facts are stored at the epoch; readers and temporal filters
# must treat this value as "unknown", never as a real 1970 date.
TS_UNKNOWN = datetime(1970, 1, 1)

_LEGACY_FORMATS = (
    "%Y/%m/%d (%a) %H:%M",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
)


def parse_datetime(
    value: DateLike,
    *,
    allow_legacy: bool = True,
    default: datetime | None = None,
) -> datetime | None:
    """Parse a datetime-like value.

    Accepts datetime objects, date objects, ISO8601 strings, and, when
    ``allow_legacy`` is true, the slash-date formats used by older tests and
    benchmark adapters. Timezone-aware inputs stay aware; naive inputs stay
    naive.
    """
    if value is None:
        return default
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if not isinstance(value, str):
        return default

    text = value.strip()
    if not text:
        return default

    iso_text = text
    if iso_text.endswith("Z"):
        iso_text = iso_text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(iso_text)
    except ValueError:
        pass

    if allow_legacy:
        for fmt in _LEGACY_FORMATS:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue

    return default


def is_unknown_ts(dt: datetime | None) -> bool:
    """True when `dt` is the ``TS_UNKNOWN`` storage sentinel (in any timezone)."""
    if dt is None:
        return False
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt == TS_UNKNOWN


def to_iso8601(dt: DateLike) -> str | None:
    """Serialize a datetime-like value to canonical ISO8601 seconds precision."""
    parsed = parse_datetime(dt)
    if parsed is None:
        return None
    return parsed.isoformat(timespec="seconds")


def format_prompt_date(dt: DateLike) -> str:
    """Format a timestamp for compact, stable reader/distiller prompts."""
    parsed = parse_datetime(dt)
    if parsed is None:
        return ""
    if parsed.tzinfo is None and parsed.timetz().replace(tzinfo=None) == time.min:
        return parsed.date().isoformat()
    return parsed.isoformat(timespec="seconds")


def datetime_sort_key(dt: DateLike) -> float:
    """Return a sortable numeric key for naive/aware datetime mixtures."""
    parsed = parse_datetime(dt)
    if parsed is None:
        return float("-inf")
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).timestamp()
    epoch = datetime(1970, 1, 1)
    return (parsed - epoch).total_seconds()


def day_range(dt: DateLike) -> tuple[datetime, datetime] | None:
    """Return [start_of_day, next_day) preserving timezone awareness."""
    parsed = parse_datetime(dt)
    if parsed is None:
        return None
    start = datetime.combine(parsed.date(), time.min, tzinfo=parsed.tzinfo)
    return start, start + timedelta(days=1)
