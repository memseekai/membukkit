"""FactDistiller failure semantics — failures raise and are never cached.

A transient LLM error used to be cached as an empty fact list keyed by content
hash, permanently losing that session's facts (re-runs read the cached [] and
never retried). These tests pin the fixed behaviour.
"""

from __future__ import annotations

import json

import pytest

from membukkit.extraction.distiller import DistillationError, FactDistiller


class FlakyLLM:
    """Fails the first `fail_n` calls, then answers."""

    def __init__(self, fail_n: int, answer: str = "0 | alice likes tea"):
        self.fail_n = fail_n
        self.calls = 0
        self.answer = answer

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        if self.calls <= self.fail_n:
            raise RuntimeError("transient LLM error")
        return self.answer


def test_distill_retries_transient_failure():
    llm = FlakyLLM(fail_n=1)
    d = FactDistiller(llm)
    facts = d.distill("k1", "user: I like tea", "2024-06-01")
    assert facts == [[0, "alice likes tea"]]
    assert llm.calls == 2  # first attempt failed, retry succeeded


def test_distill_failure_raises_and_is_not_cached():
    llm = FlakyLLM(fail_n=99)
    d = FactDistiller(llm)
    with pytest.raises(DistillationError):
        d.distill("k1", "user: I like tea", "2024-06-01")
    assert not d._cache, "a failed session must never be cached"

    # The LLM recovers -> the same session distills fine on retry.
    llm.fail_n = 0
    facts = d.distill("k1", "user: I like tea", "2024-06-01")
    assert facts == [[0, "alice likes tea"]]


def test_distill_genuinely_empty_result_is_cached():
    llm = FlakyLLM(fail_n=0, answer="NONE")
    d = FactDistiller(llm)
    assert d.distill("k1", "user: hmm", "2024-06-01") == []
    assert d.distill("k1", "user: hmm", "2024-06-01") == []
    assert llm.calls == 1, "'no facts' is a valid result and stays cached"


def test_warm_leaves_failed_sessions_uncached():
    class PerKeyLLM:
        def __call__(self, prompt: str) -> str:
            if "boom" in prompt:
                raise RuntimeError("hard failure")
            return "0 | fact"

    d = FactDistiller(PerKeyLLM())
    jobs = [("good", "user: fine", "2024-06-01"), ("bad", "user: boom", "2024-06-01")]
    d.warm(jobs, workers=2)
    assert d._vkey("good") in d._cache
    assert d._vkey("bad") not in d._cache, "failed warm jobs must stay retryable"


def test_warm_checkpoints_incrementally(tmp_path):
    """warm() persists progress mid-run, not just at the end."""
    cache = tmp_path / "distill.json"
    d = FactDistiller(lambda p: "0 | fact", cache_path=str(cache))
    jobs = [(f"k{i}", f"user: line {i}", "2024-06-01") for i in range(10)]

    d.warm(jobs, workers=4, save_every=2, save_interval_s=1e9)

    on_disk = json.loads(cache.read_text())
    assert len(on_disk) == 10
    assert all(d._vkey(f"k{i}") in on_disk for i in range(10))


def test_warm_saves_progress_on_interrupt(tmp_path):
    """A KeyboardInterrupt mid-run still flushes distilled sessions to disk."""
    cache = tmp_path / "distill.json"

    calls = {"n": 0}

    def llm(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] >= 3:
            raise KeyboardInterrupt
        return "0 | fact"

    d = FactDistiller(llm, cache_path=str(cache))
    jobs = [(f"k{i}", f"user: line {i}", "2024-06-01") for i in range(20)]

    with pytest.raises(KeyboardInterrupt):
        # workers=1 keeps completion order deterministic for the assertion.
        d.warm(jobs, workers=1)

    on_disk = json.loads(cache.read_text())
    # At least the sessions that completed before the interrupt are persisted.
    assert len(on_disk) >= 2
    # Resuming redoes only the unfinished work (content-addressed cache hits).
    assert all(v in d._cache for v in on_disk)


def test_drop_empty_entries_repairs_poisoned_cache(tmp_path):
    cache = tmp_path / "distill.json"
    poisoned = {
        "v3:failed-session": [],  # legacy cached failure
        "v3:good-session": [[0, "alice likes tea"]],
    }
    cache.write_text(json.dumps(poisoned))

    d = FactDistiller(lambda p: "NONE", cache_path=str(cache))
    assert d.drop_empty_entries() == 1
    assert "v3:good-session" in d._cache
    assert "v3:failed-session" not in d._cache
    # repair persisted to disk
    assert "failed-session" not in cache.read_text()
