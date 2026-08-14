"""Grounded Ask chip generation from facts."""

from __future__ import annotations

import json

from membukkit.suggestions import (
    _extract_json_array,
    _normalize_questions,
    refresh_store_suggestions,
    suggest_questions,
)


class _FakeBackend:
    def __init__(self, facts):
        self._facts = facts

    def count(self):
        return len(self._facts)

    def facts_page(self, offset=0, limit=50, kind=None, bucket=None):
        rows = self._facts
        if kind:
            rows = [f for f in rows if f.get("kind") == kind]
        window = rows[offset : offset + limit]
        return {"total": len(rows), "offset": offset, "facts": window}


def test_extract_json_array_fenced():
    raw = 'Here you go:\n```json\n["How much is rent?", "When did it change?"]\n```'
    assert _extract_json_array(raw) == ["How much is rent?", "When did it change?"]


def test_normalize_drops_vague():
    out = _normalize_questions(
        [
            "What are the most important facts?",
            "How much is rent as of May?",
            "How much is rent as of May?",
            "x" * 200,
        ],
        n=5,
    )
    assert out == ["How much is rent as of May?"]


def test_suggest_empty_backend():
    assert suggest_questions(_FakeBackend([]), lambda p: "[]") == []


def test_suggest_questions_ok():
    facts = [
        {
            "text": "Rent is 800€.",
            "timestamp": "2024-01-08T00:00:00",
            "kind": "atomic",
            "status": "current",
        },
        {
            "text": "Rent raised to 950€ from June.",
            "timestamp": "2024-04-02T00:00:00",
            "kind": "atomic",
            "status": "current",
        },
        {
            "text": "Gym days are Mon/Wed.",
            "timestamp": "2024-02-01T00:00:00",
            "kind": "atomic",
            "status": "current",
        },
    ]
    backend = _FakeBackend(facts)

    def llm(prompt: str) -> str:
        assert "Rent is 800" in prompt
        return json.dumps(
            [
                "How much is rent?",
                "When did rent change?",
                "Which days is the gym?",
            ]
        )

    qs = suggest_questions(backend, llm, n=5)
    assert len(qs) == 3
    assert qs[0].endswith("?")


def test_suggest_malformed_returns_empty():
    facts = [
        {
            "text": "Hello",
            "timestamp": "2024-01-01",
            "kind": "atomic",
            "status": "current",
        }
    ]
    assert suggest_questions(_FakeBackend(facts), lambda p: "not json") == []


def test_refresh_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMBUKKIT_HOME", str(tmp_path))
    from membukkit.storage.localstore import LocalStore

    store = LocalStore("chips", create=True)

    class Mem:
        backend = _FakeBackend(
            [
                {
                    "text": "Lease rent is 800€.",
                    "timestamp": "2024-01-08",
                    "kind": "atomic",
                    "status": "current",
                },
                {
                    "text": "Landlord contact is Sam.",
                    "timestamp": "2024-01-09",
                    "kind": "atomic",
                    "status": "current",
                },
                {
                    "text": "Deposit was 1600€.",
                    "timestamp": "2024-01-10",
                    "kind": "atomic",
                    "status": "current",
                },
            ]
        )
        _llm_fn = staticmethod(
            lambda p: json.dumps(
                ["How much is rent?", "Who is the landlord?", "What was the deposit?"]
            )
        )

    qs = refresh_store_suggestions(store, Mem(), llm_spec="openai:gpt-4o-mini")
    assert len(qs) == 3
    assert store.meta().get("suggested_questions") == qs
