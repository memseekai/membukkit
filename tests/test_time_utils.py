"""Tests for ISO8601 timestamp normalization helpers."""

from __future__ import annotations

from datetime import date, datetime, timezone

from membukkit.time_utils import (
    datetime_sort_key,
    format_prompt_date,
    parse_datetime,
    to_iso8601,
)


def test_parse_datetime_accepts_iso8601_variants():
    assert parse_datetime("2024-06-01") == datetime(2024, 6, 1)
    assert parse_datetime("2024-06-01T10:30:00") == datetime(2024, 6, 1, 10, 30)
    zulu = parse_datetime("2024-06-01T10:30:00Z")
    assert zulu == datetime(2024, 6, 1, 10, 30, tzinfo=timezone.utc)
    offset = parse_datetime("2024-06-01T10:30:00-05:00")
    assert offset is not None
    off = offset.utcoffset()
    assert off is not None
    assert off.total_seconds() == -5 * 3600


def test_parse_datetime_accepts_legacy_when_enabled():
    assert parse_datetime("2024/06/01") == datetime(2024, 6, 1)
    assert parse_datetime("2024/06/01 10:30") == datetime(2024, 6, 1, 10, 30)
    assert parse_datetime("2024/06/01", allow_legacy=False) is None


def test_parse_datetime_empty_and_date_objects():
    assert parse_datetime("") is None
    assert parse_datetime(None) is None
    assert parse_datetime(date(2024, 6, 1)) == datetime(2024, 6, 1)


def test_iso_and_prompt_formatting():
    assert to_iso8601(datetime(2024, 6, 1, 1, 2, 3, 999)) == "2024-06-01T01:02:03"
    assert format_prompt_date(datetime(2024, 6, 1)) == "2024-06-01"
    assert format_prompt_date(datetime(2024, 6, 1, 10, 30)) == "2024-06-01T10:30:00"
    assert format_prompt_date("2024-06-01T10:30:00Z") == "2024-06-01T10:30:00+00:00"


def test_sort_key_handles_mixed_naive_and_aware_datetimes():
    values = [
        parse_datetime("2024-06-01T10:00:00Z"),
        parse_datetime("2024-06-01T08:00:00"),
        parse_datetime("2024-06-01T09:00:00-05:00"),
    ]
    sorted_values = sorted(values, key=datetime_sort_key)
    assert sorted_values[0] == datetime(2024, 6, 1, 8)
    last = sorted_values[-1]
    assert last is not None
    off = last.utcoffset()
    assert off is not None
    assert off.total_seconds() == -5 * 3600
