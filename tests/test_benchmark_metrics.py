"""Tests for the benchmark metric and dedup logic.

The metric functions are exercised directly, not mocked: these are the numbers
the benchmarks report, so a bug here would silently misstate results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from benchmarks.common import qmd_compat
from benchmarks.common.dedup import chunk_span_by_document, collapse_to_documents
from benchmarks.common.metrics import (
    all_support_at_k,
    any_support_at_k,
    first_relevant_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


@dataclass
class Hit:
    doc_id: str = ""
    doc_name: str = ""


# ------------------------------------------------------- chunk -> document
def test_collapse_keeps_first_occurrence_rank():
    hits = [Hit("a"), Hit("a"), Hit("b"), Hit("a"), Hit("c")]
    assert collapse_to_documents(hits) == ["a", "b", "c"]


def test_collapse_does_not_reorder_by_frequency():
    """A document hit once but first must still outrank one hit three times."""
    hits = [Hit("rare"), Hit("common"), Hit("common"), Hit("common")]
    assert collapse_to_documents(hits)[0] == "rare"


def test_collapse_falls_back_to_doc_name():
    assert collapse_to_documents([Hit(doc_id="", doc_name="x.md")]) == ["x.md"]


def test_collapse_skips_hits_with_no_document_id():
    """Unidentifiable hits must not merge into one phantom document."""
    hits = [Hit(""), Hit("a"), Hit("")]
    assert collapse_to_documents(hits) == ["a"]


def test_chunk_span_counts_contributions():
    hits = [Hit("a"), Hit("a"), Hit("b")]
    assert chunk_span_by_document(hits) == {"a": 2, "b": 1}


# ------------------------------------------------------------------ recall
def test_recall_at_k_is_fraction_of_gold_found():
    ranked = ["a", "x", "b"]
    assert recall_at_k(ranked, ["a", "b"], 3) == 1.0
    assert recall_at_k(ranked, ["a", "b"], 1) == 0.5
    assert recall_at_k(ranked, ["a", "b"], 2) == 0.5


def test_recall_ignores_duplicates_in_gold():
    assert recall_at_k(["a"], ["a", "a"], 1) == 1.0


def test_recall_with_no_gold_is_zero_not_error():
    assert recall_at_k(["a"], [], 5) == 0.0


# -------------------------------------------------- any / all support
def test_any_support_needs_only_one_gold_document():
    assert any_support_at_k(["x", "a"], ["a", "b"], 2) == 1.0
    assert any_support_at_k(["x", "y"], ["a", "b"], 2) == 0.0


def test_all_support_needs_every_gold_document():
    assert all_support_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    assert all_support_at_k(["a", "x"], ["a", "b"], 2) == 0.0


def test_all_support_is_stricter_than_any_for_multi_gold():
    """The distinction that makes multi-hop interesting."""
    ranked, gold = ["a", "x", "y"], ["a", "b"]
    assert any_support_at_k(ranked, gold, 3) == 1.0
    assert all_support_at_k(ranked, gold, 3) == 0.0


def test_all_support_respects_the_k_cutoff():
    ranked, gold = ["a", "x", "b"], ["a", "b"]
    assert all_support_at_k(ranked, gold, 2) == 0.0
    assert all_support_at_k(ranked, gold, 3) == 1.0


def test_single_gold_collapses_the_three_families():
    ranked, gold = ["x", "a"], ["a"]
    assert recall_at_k(ranked, gold, 2) == any_support_at_k(ranked, gold, 2) == 1.0
    assert all_support_at_k(ranked, gold, 2) == 1.0


# --------------------------------------------------------------------- MRR
def test_reciprocal_rank_uses_first_relevant_position():
    assert reciprocal_rank(["a"], ["a"]) == 1.0
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_nothing_relevant_retrieved():
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_reciprocal_rank_takes_the_earliest_of_several_gold():
    assert reciprocal_rank(["x", "b", "a"], ["a", "b"]) == 0.5


def test_first_relevant_rank_is_one_based_or_none():
    assert first_relevant_rank(["a"], ["a"]) == 1
    assert first_relevant_rank(["x", "y"], ["a"]) is None


# -------------------------------------------------------- precision / nDCG
def test_precision_at_k_divides_by_k():
    assert precision_at_k(["a", "x"], ["a"], 2) == 0.5


def test_ndcg_rewards_earlier_placement():
    early = ndcg_at_k(["a", "x", "y"], ["a"], 3)
    late = ndcg_at_k(["x", "y", "a"], ["a"], 3)
    assert early == 1.0
    assert late < early


def test_ndcg_perfect_ordering_for_multi_gold_is_one():
    assert ndcg_at_k(["a", "b", "x"], ["a", "b"], 3) == pytest.approx(1.0)


# ----------------------------------------------------------- QMD semantics
def test_qmd_normalize_strips_scheme_and_collection():
    assert qmd_compat.normalize_path("qmd://eval-docs/docs/readme.md") == "docs/readme.md"
    assert qmd_compat.normalize_path("/Foo.MD/") == "foo.md"


def test_qmd_paths_match_on_suffix_either_direction():
    assert qmd_compat.paths_match("a/b/api-design-principles.md", "api-design-principles.md")
    assert qmd_compat.paths_match("api-design-principles.md", "a/b/api-design-principles.md")
    assert not qmd_compat.paths_match("other.md", "api-design-principles.md")


def test_qmd_precision_denominator_is_min_k_and_expected():
    """QMD divides by min(k, |expected|), so one hit at k=10 scores 1.0."""
    assert qmd_compat.qmd_precision_at_k(["a"] + ["x"] * 9, ["a"], 10) == 1.0
    assert precision_at_k(["a"] + ["x"] * 9, ["a"], 10) == pytest.approx(0.1)


def test_qmd_recall_is_over_all_results_not_top_k():
    """Found at rank 9 still counts as recall 1.0 in QMD's scorer."""
    results = ["x"] * 8 + ["a"]
    assert qmd_compat.score_results(results, ["a"], top_k=3)["recall"] == 1.0
    assert qmd_compat.score_results(results, ["a"], top_k=3)["recall_at_3"] == 0.0


def test_qmd_score_results_shape_and_mrr():
    out = qmd_compat.score_results(["x", "a"], ["a"], top_k=5)
    assert out["mrr"] == 0.5
    assert out["matched_files"] == ["a"]
    assert out["unmatched_expected_files"] == []
    assert out["hits_at_k"] == 1


# ------------------------------------------------- HotpotQA corpus + sampling
def _hf_record(qid: str, gold=("A", "B")):
    """A row shaped like Hugging Face's hotpot_qa parquet (columnar)."""
    titles = ["A", "B", "C", "D"]
    return {
        "id": qid,
        "question": f"question {qid}?",
        "answer": "SECRET-ANSWER",
        "type": "bridge",
        "level": "hard",
        "supporting_facts": {"title": list(gold), "sent_id": [0] * len(gold)},
        "context": {"title": titles, "sentences": [[f"Sentence about {t}. "] for t in titles]},
    }


def test_render_document_is_deterministic_and_verbatim():
    from benchmarks.hotpotqa.dataset import render_document

    a = render_document("Tokyo", ["Tokyo is a city. ", "It is large."])
    assert a == render_document("Tokyo", ["Tokyo is a city. ", "It is large."])
    assert a.startswith("# Tokyo\n\n")
    assert "Tokyo is a city. It is large." in a


def test_stable_doc_id_is_scoped_per_question():
    from benchmarks.hotpotqa.dataset import stable_doc_id

    assert stable_doc_id("q1", "Tokyo") == stable_doc_id("q1", "Tokyo")
    assert stable_doc_id("q1", "Tokyo") != stable_doc_id("q2", "Tokyo")


def test_parses_hugging_face_columnar_schema():
    from benchmarks.hotpotqa.dataset import question_from_record

    q = question_from_record(_hf_record("q1"))
    assert q.qid == "q1"
    assert q.gold_titles == ["A", "B"]
    assert [d["title"] for d in q.docs] == ["A", "B", "C", "D"]
    assert q.docs[0]["text"].startswith("# A\n\n")


def test_deterministic_sampling():
    from benchmarks.hotpotqa.dataset import questions_from_records

    recs = [_hf_record(f"q{i:03d}") for i in range(50)]
    a = questions_from_records(recs, limit=10, seed=42)
    b = questions_from_records(recs, limit=10, seed=42)
    c = questions_from_records(recs, limit=10, seed=7)
    assert [q.qid for q in a] == [q.qid for q in b]
    assert [q.qid for q in a] != [q.qid for q in c]
    assert len(a) == 10


def test_sampling_is_order_independent():
    """Shuffling the source file must not change which questions are sampled."""
    import random as _r
    from benchmarks.hotpotqa.dataset import questions_from_records

    recs = [_hf_record(f"q{i:03d}") for i in range(50)]
    shuffled = recs[:]
    _r.Random(1).shuffle(shuffled)
    assert [q.qid for q in questions_from_records(recs, limit=10, seed=42)] == [
        q.qid for q in questions_from_records(shuffled, limit=10, seed=42)
    ]


def test_gold_labels_never_enter_document_text():
    """Answers and supporting-fact labels must not be indexable."""
    from benchmarks.hotpotqa.dataset import question_from_record

    q = question_from_record(_hf_record("q1"))
    blob = " ".join(d["text"] for d in q.docs) + " " + " ".join(d["doc_id"] for d in q.docs)
    assert "SECRET-ANSWER" not in blob
    assert "supporting" not in blob.lower()
    assert q.gold_titles == ["A", "B"]


def test_gold_doc_ids_map_titles_to_this_questions_docs():
    from benchmarks.hotpotqa.dataset import gold_doc_ids, question_from_record

    q = question_from_record(_hf_record("q1"))
    gold = gold_doc_ids(q)
    assert len(gold) == 2
    assert set(gold) <= {d["doc_id"] for d in q.docs}
