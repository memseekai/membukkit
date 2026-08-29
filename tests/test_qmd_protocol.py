"""Tests for the QMD-protocol report builder and exported filename derivation.

The report schema is checked field-for-field against a real ``qmd bench --json``
block, because the whole value of this path is that the two files are diffable.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.common import qmd_report
from benchmarks.common.paths import slugify, unique_filename

# Field names taken verbatim from a real `qmd bench --json` backend block.
QMD_BACKEND_FIELDS = {
    "precision_at_k", "recall", "recall_at_1", "recall_at_3", "recall_at_5",
    "mrr", "f1", "hits_at_k", "matched_files", "unmatched_expected_files",
    "total_expected", "latency_ms", "top_files",
}
QMD_SUMMARY_FIELDS = {
    "avg_precision", "avg_recall", "avg_recall_at_1", "avg_recall_at_3",
    "avg_recall_at_5", "avg_mrr", "avg_f1", "avg_latency_ms",
}


# ------------------------------------------------------------------ slugify
def test_slugify_strips_leading_dots_so_no_dotfiles_are_written():
    """Dotfiles are skipped by most indexers, which would shrink one corpus."""
    assert slugify(".hack//Sign") == "hack-sign"
    assert slugify(".50 BMG") == "50-bmg"
    assert not slugify("...Baby One More Time").startswith(".")


def test_slugify_lowercases_and_replaces_unsafe_characters():
    assert slugify("Blade Runner: 2049!") == "blade-runner-2049"


def test_slugify_falls_back_for_titles_with_nothing_usable():
    assert slugify("///") == "untitled"
    assert slugify("") == "untitled"


def test_slugify_truncates_long_titles():
    assert len(slugify("x" * 500)) == 120


def test_unique_filename_disambiguates_repeated_titles():
    used = {}
    assert unique_filename("Dup", used) == "dup.md"
    assert unique_filename("Dup", used) == "dup-1.md"
    assert unique_filename("Dup", used) == "dup-2.md"


def test_unique_filename_avoids_colliding_with_a_real_title():
    """A generated 'foo-1' must not collide with a title that slugifies to 'foo-1'."""
    used = {}
    assert unique_filename("Foo", used) == "foo.md"
    assert unique_filename("Foo 1", used) == "foo-1.md"
    third = unique_filename("Foo", used)
    assert third not in ("foo.md", "foo-1.md")


def test_unique_filename_never_repeats_across_many_titles():
    used = {}
    titles = ["Foo", "Foo 1", "Foo", "Foo 2", "Foo", "Foo-1", "Foo"]
    names = [unique_filename(t, used) for t in titles]
    assert len(names) == len(set(names))


# ------------------------------------------------------------ report schema
QUERIES = [
    {"id": "q1", "query": "who?", "type": "2hop",
     "expected_files": ["a.md", "b.md"], "expected_in_top_k": 5},
    {"id": "q2", "query": "when?", "type": "3hop",
     "expected_files": ["c.md"], "expected_in_top_k": 5},
]


def _runs(**by_query):
    return {qid: {"top_files": files, "latency_ms": 10.0}
            for qid, files in by_query.items()}


def test_backend_block_matches_qmd_field_set():
    block = qmd_report.score_backend(["a.md", "x.md"], ["a.md", "b.md"], 5, 12.0)
    assert set(block) == QMD_BACKEND_FIELDS


def test_summary_block_matches_qmd_field_set():
    report = qmd_report.build_report(
        "f.json", QUERIES,
        {"chain": _runs(q1=["a.md", "b.md"], q2=["c.md"])})
    assert set(report["summary"]["chain"]) == QMD_SUMMARY_FIELDS


def test_report_has_qmd_top_level_shape():
    report = qmd_report.build_report("f.json", QUERIES, {"chain": _runs(q1=["a.md"])})
    assert set(report) >= {"timestamp", "fixture", "results", "summary"}
    assert set(report["results"][0]) == {"id", "query", "type", "backends"}


def test_scoring_uses_qmd_precision_denominator():
    """One hit at k=5 with one expected file scores 1.0, not 0.2."""
    block = qmd_report.score_backend(["c.md"] + ["x.md"] * 4, ["c.md"], 5, 1.0)
    assert block["precision_at_k"] == 1.0


def test_recall_is_over_all_results_not_top_k():
    block = qmd_report.score_backend(["x.md"] * 8 + ["a.md"], ["a.md"], 3, 1.0)
    assert block["recall"] == 1.0
    assert block["recall_at_3"] == 0.0


def test_qmd_uri_prefix_is_reported_and_still_matches_gold():
    files = [qmd_report.qmd_uri("musique", "a.md")]
    assert files == ["qmd://musique/a.md"]
    block = qmd_report.score_backend(files, ["a.md"], 5, 1.0)
    assert block["matched_files"] == ["a.md"]


def test_missing_backend_for_a_query_is_omitted_not_zeroed():
    report = qmd_report.build_report(
        "f.json", QUERIES, {"chain": _runs(q1=["a.md"])})
    assert "chain" in report["results"][0]["backends"]
    assert report["results"][1]["backends"] == {}


def test_summary_averages_over_answered_queries_only():
    report = qmd_report.build_report(
        "f.json", QUERIES, {"chain": _runs(q1=["a.md", "b.md"])})
    assert report["summary"]["chain"]["avg_recall"] == 1.0


def test_multiple_backends_are_summarised_separately():
    report = qmd_report.build_report(
        "f.json", QUERIES,
        {"dense": _runs(q1=["x.md"], q2=["c.md"]),
         "chain": _runs(q1=["a.md", "b.md"], q2=["c.md"])})
    assert report["summary"]["chain"]["avg_recall"] > report["summary"]["dense"]["avg_recall"]


def test_extra_metadata_is_merged_into_the_report():
    report = qmd_report.build_report("f.json", QUERIES, {"chain": _runs(q1=["a.md"])},
                                     extra={"system": "membukkit"})
    assert report["system"] == "membukkit"


# ------------------------------------------------------------- comparison
def _report(label_runs):
    return qmd_report.build_report("f.json", QUERIES, label_runs)


def test_shared_query_ids_intersects_reports():
    a = _report({"chain": _runs(q1=["a.md"], q2=["c.md"])})
    b = qmd_report.build_report("f.json", QUERIES[:1], {"vector": _runs(q1=["a.md"])})
    assert qmd_report.shared_query_ids([a, b]) == ["q1"]


def test_restrict_resummarises_over_the_subset():
    r = _report({"chain": _runs(q1=["x.md"], q2=["c.md"])})
    only_q2 = qmd_report.restrict(r, ["q2"])
    assert len(only_q2["results"]) == 1
    assert only_q2["summary"]["chain"]["avg_recall"] == 1.0


def test_summary_table_renders_every_backend():
    r = _report({"dense": _runs(q1=["a.md"]), "chain": _runs(q1=["a.md", "b.md"])})
    table = qmd_report.summary_table({"membukkit": r})
    assert "dense" in table and "chain" in table


def test_load_fixture_rejects_a_non_fixture(tmp_path):
    p = tmp_path / "nope.json"
    p.write_text(json.dumps({"not": "a fixture"}))
    with pytest.raises(ValueError, match="not a QMD fixture"):
        qmd_report.load_fixture(p)


def test_load_fixture_reads_queries(tmp_path):
    p = tmp_path / "queries.json"
    p.write_text(json.dumps({"collection": "musique", "queries": QUERIES}))
    assert len(qmd_report.load_fixture(p)["queries"]) == 2


def test_duplicate_query_ids_are_rejected():
    """Guard against the collapse that silently zeroed a whole 2Wiki run."""
    dupes = [{"id": "same", "query": "a?", "expected_files": ["a.md"], "expected_in_top_k": 5},
             {"id": "same", "query": "b?", "expected_files": ["b.md"], "expected_in_top_k": 5}]
    with pytest.raises(ValueError, match="duplicate query ids"):
        qmd_report.build_report("f.json", dupes, {"chain": _runs(same=["a.md"])})


# ------------------------------------------------- config snapshot (regression)
def test_config_snapshot_records_model_overrides():
    """Results must say which encoder and reranker produced them."""
    from benchmarks.multihop.run_fixture import _config_snapshot

    snap = _config_snapshot("Qwen/Qwen3-Embedding-0.6B", 384, 8, "", 10,
                            reranker_path="/models/reranker-rag-v1")
    assert snap["encoder"] == "Qwen/Qwen3-Embedding-0.6B"
    assert snap["encoder_overridden"] is True
    assert snap["reranker"] == "/models/reranker-rag-v1"
    assert snap["reranker_overridden"] is True


def test_config_snapshot_falls_back_to_shipped_models():
    from benchmarks.multihop.run_fixture import _config_snapshot

    snap = _config_snapshot("", 384, 8, "", 10)
    assert snap["encoder_overridden"] is False
    assert snap["reranker_overridden"] is False
    assert snap["encoder"] and snap["reranker"]
