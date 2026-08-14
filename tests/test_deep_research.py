"""Tests for memory DeepSearch deterministic control logic."""

from __future__ import annotations

from membukkit.agents.deep_research import (
    DeepSearchState,
    EvidenceNote,
    allowed_actions,
    dedupe_queries,
    rotate_gap_questions,
    validate_citations,
)


def test_dedupe_queries_drops_near_duplicates():
    queries = [
        "JDC Barcelona meetings",
        "jdc   barcelona meetings",
        "JDC Barcelona dinner",
    ]
    assert dedupe_queries(queries) == ["JDC Barcelona meetings", "JDC Barcelona dinner"]


def test_rotate_gap_questions_pushes_gaps_before_original():
    state = DeepSearchState(original_question="What happened?", gap_queue=["old gap"])
    added = rotate_gap_questions(state, ["Who was involved?", "When was it?"])

    assert added == ["Who was involved?", "When was it?"]
    assert state.gap_queue[:3] == ["Who was involved?", "When was it?", "old gap"]
    assert state.gap_queue[-1] == "What happened?"


def test_allowed_actions_gates_answer_until_evidence_and_after_failure():
    state = DeepSearchState(original_question="q", min_evidence=2)
    assert allowed_actions(state) == ["reflect", "search"]

    state.knowledge_ledger = [
        EvidenceNote(ref="mem:1", fact="a", text="a", query="q"),
        EvidenceNote(ref="mem:2", fact="b", text="b", query="q"),
    ]
    assert allowed_actions(state) == ["reflect", "search", "answer"]

    state.last_answer_failed = True
    assert allowed_actions(state) == ["reflect", "search"]


def test_validate_citations_splits_valid_and_invalid_refs():
    evidence = [
        EvidenceNote(ref="mem:aaa", fact="a", text="a", query="q"),
        EvidenceNote(ref="mem:bbb", fact="b", text="b", query="q"),
    ]

    valid, invalid = validate_citations(["mem:aaa", "mem:nope", "mem:aaa"], evidence)
    assert valid == ["mem:aaa"]
    assert invalid == ["mem:nope"]
