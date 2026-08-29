"""Tests for the shared-corpus multi-hop split loader.

Fixtures are written to tmp_path and loaded with ``strict=False``, so nothing
here downloads anything. The size guardrail itself is tested separately.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.multihop import dataset as ds


def _write(tmp_path, qa, corpus, name="musique"):
    qa_name, corpus_name = ds._FILES[ds.ALIASES[name]]
    (tmp_path / qa_name).write_text(json.dumps(qa))
    (tmp_path / corpus_name).write_text(json.dumps(corpus))
    return tmp_path


CORPUS = [
    {"title": "Film X", "text": "Directed by Jane Roe."},
    {"title": "Jane Roe", "text": "Won the Palme dOr."},
    {"title": "Lentils", "text": "A legume."},
]


def _musique_q(qid="2hop__1", supporting=("Film X", "Jane Roe")):
    return {
        "id": qid,
        "question": "What award did the director of Film X win?",
        "answer": "Palme dOr",
        "answer_aliases": ["the Palme d'Or"],
        "paragraphs": [
            {"title": t, "paragraph_text": "...", "is_supporting": t in supporting}
            for t in ("Film X", "Jane Roe", "Lentils")
        ],
    }


# ------------------------------------------------------------------ parsing
def test_loads_questions_and_shared_corpus(tmp_path):
    d = _write(tmp_path, [_musique_q()], CORPUS)
    s = ds.load_split("musique", cache_dir=d, strict=False)
    assert len(s.questions) == 1
    assert len(s.passages) == 3


def test_passage_text_matches_hipporag_format():
    assert ds.passage_text("T", "body") == "T\nbody"


def test_passage_text_is_used_verbatim(tmp_path):
    d = _write(tmp_path, [_musique_q()], CORPUS)
    s = ds.load_split("musique", cache_dir=d, strict=False)
    assert s.passages[0].text == "Film X\nDirected by Jane Roe."


def test_gold_titles_from_is_supporting(tmp_path):
    d = _write(tmp_path, [_musique_q()], CORPUS)
    s = ds.load_split("musique", cache_dir=d, strict=False)
    assert s.questions[0].gold_titles == ["Film X", "Jane Roe"]


def test_gold_titles_from_supporting_facts_format(tmp_path):
    q = {"id": "x1", "question": "q?",
         "supporting_facts": [["Film X", 0], ["Jane Roe", 1], ["Film X", 2]]}
    d = _write(tmp_path, [q], CORPUS, name="hotpot")
    s = ds.load_split("hotpot", cache_dir=d, strict=False)
    assert s.questions[0].gold_titles == ["Film X", "Jane Roe"]


def test_hop_type_parsed_from_musique_id(tmp_path):
    d = _write(tmp_path, [_musique_q("3hop1__42")], CORPUS)
    s = ds.load_split("musique", cache_dir=d, strict=False)
    assert s.questions[0].hop_type == "3hop"


def test_hop_type_defaults_to_multihop(tmp_path):
    d = _write(tmp_path, [_musique_q("plain-id")], CORPUS)
    s = ds.load_split("musique", cache_dir=d, strict=False)
    assert s.questions[0].hop_type == "multihop"


def test_questions_without_gold_are_dropped(tmp_path):
    good = _musique_q("2hop__1")
    bad = _musique_q("2hop__2", supporting=())
    d = _write(tmp_path, [good, bad], CORPUS)
    s = ds.load_split("musique", cache_dir=d, strict=False)
    assert [q.qid for q in s.questions] == ["2hop__1"]


# ----------------------------------------------------------------- sampling
def test_sampling_is_deterministic(tmp_path):
    qa = [_musique_q(f"2hop__{i}") for i in range(50)]
    d = _write(tmp_path, qa, CORPUS)
    a = ds.load_split("musique", limit=10, seed=7, cache_dir=d, strict=False)
    b = ds.load_split("musique", limit=10, seed=7, cache_dir=d, strict=False)
    assert [q.qid for q in a.questions] == [q.qid for q in b.questions]


def test_different_seeds_select_different_questions(tmp_path):
    qa = [_musique_q(f"2hop__{i}") for i in range(50)]
    d = _write(tmp_path, qa, CORPUS)
    a = ds.load_split("musique", limit=10, seed=1, cache_dir=d, strict=False)
    b = ds.load_split("musique", limit=10, seed=2, cache_dir=d, strict=False)
    assert [q.qid for q in a.questions] != [q.qid for q in b.questions]


def test_sampling_is_independent_of_source_file_order(tmp_path):
    """Shuffling the source file must not change which questions are sampled."""
    qa = [_musique_q(f"2hop__{i}") for i in range(50)]
    forward, reverse = tmp_path / "forward", tmp_path / "reverse"
    forward.mkdir()
    reverse.mkdir()
    _write(forward, qa, CORPUS)
    _write(reverse, list(reversed(qa)), CORPUS)

    a = ds.load_split("musique", limit=10, seed=3, cache_dir=forward, strict=False)
    b = ds.load_split("musique", limit=10, seed=3, cache_dir=reverse, strict=False)
    assert [q.qid for q in a.questions] == [q.qid for q in b.questions]


def test_corpus_is_never_sampled(tmp_path):
    qa = [_musique_q(f"2hop__{i}") for i in range(50)]
    d = _write(tmp_path, qa, CORPUS)
    s = ds.load_split("musique", limit=5, cache_dir=d, strict=False)
    assert len(s.questions) == 5
    assert len(s.passages) == len(CORPUS)


def test_limit_larger_than_split_returns_everything(tmp_path):
    d = _write(tmp_path, [_musique_q()], CORPUS)
    s = ds.load_split("musique", limit=999, cache_dir=d, strict=False)
    assert len(s.questions) == 1


# --------------------------------------------------------------- guardrails
def test_strict_mode_rejects_a_wrong_sized_split(tmp_path):
    d = _write(tmp_path, [_musique_q()], CORPUS)
    with pytest.raises(AssertionError, match="size mismatch"):
        ds.load_split("musique", cache_dir=d, strict=True)


def test_unknown_dataset_rejected():
    with pytest.raises(ValueError, match="unknown dataset"):
        ds.load_split("trivia-night")


def test_aliases_resolve_to_official_names(tmp_path):
    d = _write(tmp_path, [_musique_q()], CORPUS, name="2wiki")
    s = ds.load_split("2wiki", cache_dir=d, strict=False)
    assert s.name == "2wikimultihopqa"


def test_split_records_checksums(tmp_path):
    d = _write(tmp_path, [_musique_q()], CORPUS)
    s = ds.load_split("musique", cache_dir=d, strict=False)
    assert len(s.qa_sha256) == 64
    assert len(s.corpus_sha256) == 64


# ------------------------------------------------------------- doc id / gold
def test_doc_ids_are_unique_even_with_repeated_titles(tmp_path):
    corpus = [{"title": "Dup", "text": "one"}, {"title": "Dup", "text": "two"}]
    d = _write(tmp_path, [_musique_q()], corpus)
    s = ds.load_split("musique", cache_dir=d, strict=False)
    assert len({p.doc_id for p in s.passages}) == 2


def test_title_of_recovers_titles_containing_colons(tmp_path):
    corpus = [{"title": "Blade Runner: 2049", "text": "x"}]
    d = _write(tmp_path, [_musique_q()], corpus)
    s = ds.load_split("musique", cache_dir=d, strict=False)
    assert ds.title_of(s.passages[0].doc_id) == "Blade Runner: 2049"


def test_gold_doc_ids_maps_titles_to_every_matching_passage(tmp_path):
    corpus = [{"title": "Film X", "text": "a"}, {"title": "Film X", "text": "b"},
              {"title": "Lentils", "text": "c"}]
    d = _write(tmp_path, [_musique_q()], corpus)
    s = ds.load_split("musique", cache_dir=d, strict=False)
    got = ds.gold_doc_ids(s.questions[0], s.passages)
    assert got == ["0:Film X", "1:Film X"]


def test_answers_never_appear_in_passage_text(tmp_path):
    d = _write(tmp_path, [_musique_q()], CORPUS)
    s = ds.load_split("musique", cache_dir=d, strict=False)
    blob = " ".join(p.text for p in s.passages)
    assert "answer_aliases" not in blob


def test_gold_labels_never_appear_in_searchable_fields(tmp_path):
    d = _write(tmp_path, [_musique_q()], CORPUS)
    s = ds.load_split("musique", cache_dir=d, strict=False)
    for p in s.passages:
        assert "is_supporting" not in p.text


# --------------------------------------------- question id robustness (regression)
def test_reads_2wiki_style_underscore_id(tmp_path):
    """2Wiki/HotpotQA use '_id'; reading only 'id' yields "None" for every row."""
    q = {"_id": "83bf3b5a", "question": "q?", "type": "compositional",
         "supporting_facts": [["Film X", 0], ["Jane Roe", 1]]}
    d = _write(tmp_path, [q], CORPUS, name="2wiki")
    s = ds.load_split("2wiki", cache_dir=d, strict=False)
    assert s.questions[0].qid == "83bf3b5a"


def test_question_id_is_never_the_string_none(tmp_path):
    q = {"question": "q?", "supporting_facts": [["Film X", 0]]}
    d = _write(tmp_path, [q], CORPUS, name="2wiki")
    s = ds.load_split("2wiki", cache_dir=d, strict=False)
    assert s.questions[0].qid not in ("None", "", None)


def test_duplicate_question_ids_fail_loudly(tmp_path):
    """N questions sharing an id collapse into one result downstream."""
    qa = [_musique_q("same"), _musique_q("same")]
    d = _write(tmp_path, qa, CORPUS)
    with pytest.raises(AssertionError, match="duplicate question id"):
        ds.load_split("musique", cache_dir=d, strict=False)


def test_hop_type_prefers_the_splits_own_label(tmp_path):
    q = {"_id": "x1", "question": "q?", "type": "comparison",
         "supporting_facts": [["Film X", 0]]}
    d = _write(tmp_path, [q], CORPUS, name="2wiki")
    s = ds.load_split("2wiki", cache_dir=d, strict=False)
    assert s.questions[0].hop_type == "comparison"


def test_every_question_gets_a_distinct_id_without_any_id_field(tmp_path):
    qa = [{"question": f"q{i}?", "supporting_facts": [["Film X", 0]]} for i in range(5)]
    d = _write(tmp_path, qa, CORPUS, name="2wiki")
    s = ds.load_split("2wiki", cache_dir=d, strict=False)
    assert len({q.qid for q in s.questions}) == 5


# ---------------------------------- gold resolves to exact passages (regression)
def _musique_dup_corpus():
    """Two passages share a title; only one is the supporting paragraph."""
    return [
        {"title": "Barca", "text": "Founded in 1899."},
        {"title": "Barca", "text": "Won the treble in 2009."},
        {"title": "Lentils", "text": "A legume."},
    ]


def test_gold_matches_the_exact_supporting_passage_not_every_same_title(tmp_path):
    q = {
        "id": "2hop__1", "question": "q?", "answer": "a",
        "paragraphs": [
            {"title": "Barca", "paragraph_text": "Won the treble in 2009.",
             "is_supporting": True},
            {"title": "Lentils", "paragraph_text": "A legume.", "is_supporting": False},
        ],
    }
    d = _write(tmp_path, [q], _musique_dup_corpus())
    s = ds.load_split("musique", cache_dir=d, strict=False)
    gold = ds.gold_doc_ids(s.questions[0], s.passages)
    assert len(gold) == 1, f"one supporting paragraph must map to one passage, got {gold}"
    assert ds.title_of(gold[0]) == "Barca"
    assert "treble" in next(p.text for p in s.passages if p.doc_id == gold[0])


def test_title_only_gold_still_resolves_when_text_is_absent(tmp_path):
    """2Wiki/HotpotQA give titles only; their corpora have unique titles."""
    q = {"_id": "x1", "question": "q?", "supporting_facts": [["Lentils", 0]]}
    d = _write(tmp_path, [q], _musique_dup_corpus(), name="2wiki")
    s = ds.load_split("2wiki", cache_dir=d, strict=False)
    assert [ds.title_of(g) for g in ds.gold_doc_ids(s.questions[0], s.passages)] == ["Lentils"]


def test_gold_count_matches_supporting_paragraph_count(tmp_path):
    q = {
        "id": "2hop__2", "question": "q?",
        "paragraphs": [
            {"title": "Barca", "paragraph_text": "Founded in 1899.", "is_supporting": True},
            {"title": "Lentils", "paragraph_text": "A legume.", "is_supporting": True},
        ],
    }
    d = _write(tmp_path, [q], _musique_dup_corpus())
    s = ds.load_split("musique", cache_dir=d, strict=False)
    assert s.questions[0].n_gold == 2
