"""Token usage metering and cost estimates."""

from __future__ import annotations

from membukkit.usage import (
    TokenUsage,
    estimate_cost_usd,
    estimate_tokens,
    format_cost,
    get_meter,
    merge_usage_into_meta,
    window_fraction,
)


def test_estimate_tokens():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_meter_take_resets():
    m = get_meter()
    m.take()
    m.add(100, 20, source="api")
    u = m.take()
    assert u.prompt_tokens == 100
    assert u.completion_tokens == 20
    assert u.source == "api"
    assert m.take().total_tokens == 0


def test_cost_and_window():
    u = TokenUsage(prompt_tokens=1_000_000, completion_tokens=0, source="api")
    assert estimate_cost_usd(u, "openai:gpt-4o-mini") == 0.15
    assert abs(window_fraction(1280) - 0.01) < 1e-9
    assert format_cost(0.0003).startswith("$")


def test_merge_usage_meta():
    meta = {}
    totals = merge_usage_into_meta(
        meta,
        TokenUsage(prompt_tokens=1000, completion_tokens=100, source="api"),
        "openai:gpt-4o-mini",
    )
    assert totals["prompt_tokens"] == 1000
    assert totals["est_cost_usd"] is not None
    meta2 = {"usage_totals": totals}
    totals2 = merge_usage_into_meta(
        meta2,
        TokenUsage(prompt_tokens=1000, completion_tokens=0, source="estimate"),
        "openai:gpt-4o-mini",
    )
    assert totals2["prompt_tokens"] == 2000
    assert totals2["source"] == "mixed"
