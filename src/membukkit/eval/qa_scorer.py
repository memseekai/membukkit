"""Official SQuAD/MRQA-style QA scorer for the multi-hop RAG benchmark.

This is VENDORED VERBATIM from the official HippoRAG repository
(OSU-NLP-Group/HippoRAG, src/hipporag/utils/eval_utils.py::normalize_answer and
src/hipporag/evaluation/qa_eval.py::QAExactMatch / QAF1Score) so that EM/F1 on
MuSiQue / 2WikiMultiHopQA / HotpotQA are computed with the *same* normalization
the published baselines (and the anchor paper) use. Do NOT hand-roll or "improve"
this normalization — byte-for-byte parity with HippoRAG's scorer is the point.

We DELIBERATELY do not reuse coremem3.eval.metrics.f1_score here: that one strips
a different article set and is the conversational-J-score token-F1, not the
SQuAD/MRQA QA scorer. Keeping them separate avoids silently changing the metric.

Gold answers are a LIST per question (answer + aliases); per-question scores are
aggregated across the gold list with max (MRQA convention), then averaged.

Provenance: HippoRAG @ github.com/OSU-NLP-Group/HippoRAG (MIT License),
normalize_answer ultimately from the SQuAD v1.1 official eval script.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Sequence


# --- VENDORED: HippoRAG src/hipporag/utils/eval_utils.py -----------------------
def normalize_answer(answer: str) -> str:
    """Lowercase, strip punctuation + articles (a/an/the), collapse whitespace.

    Verbatim from HippoRAG (SQuAD v1.1 normalization)."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(answer)))))


# --- VENDORED: HippoRAG src/hipporag/evaluation/qa_eval.py ---------------------
def _em_single(gold: str, predicted: str) -> float:
    return 1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0


def _f1_single(gold: str, predicted: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    predicted_tokens = normalize_answer(predicted).split()
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(predicted_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)


def exact_match(gold_answers: Sequence[str], predicted: str) -> float:
    """EM aggregated (max) over a list of acceptable gold answers."""
    if not gold_answers:
        return 0.0
    return max(_em_single(g, predicted) for g in gold_answers)


def f1(gold_answers: Sequence[str], predicted: str) -> float:
    """Token-F1 aggregated (max) over a list of acceptable gold answers."""
    if not gold_answers:
        return 0.0
    return max(_f1_single(g, predicted) for g in gold_answers)


def score_qa(
    gold_answers: Sequence[Sequence[str]],
    predictions: Sequence[str],
) -> Dict[str, float]:
    """Corpus-level EM/F1 (percent), MRQA aggregation. Lengths must match."""
    assert len(gold_answers) == len(predictions), (
        f"gold ({len(gold_answers)}) vs pred ({len(predictions)}) length mismatch"
    )
    n = len(predictions)
    if n == 0:
        return {"em": 0.0, "f1": 0.0, "n": 0}
    em = sum(exact_match(g, p) for g, p in zip(gold_answers, predictions)) / n
    f1_ = sum(f1(g, p) for g, p in zip(gold_answers, predictions)) / n
    return {"em": 100.0 * em, "f1": 100.0 * f1_, "n": n}


# --- Passage retrieval recall (HippoRAG retrieval_eval convention) -------------
def recall_at_k(
    retrieved_titles: Sequence[str],
    gold_titles: Sequence[str],
    k: int,
) -> float:
    """Fraction of gold supporting passages present in the top-k retrieved.

    Matches HippoRAG's retrieval recall: |gold ∩ top_k| / |gold|. Passage identity
    is the (unique) passage title in these corpora.
    """
    gold = set(gold_titles)
    if not gold:
        return 0.0
    topk = set(retrieved_titles[:k])
    return len(gold & topk) / len(gold)


def score_retrieval(
    retrieved_titles: Sequence[Sequence[str]],
    gold_titles: Sequence[Sequence[str]],
    ks: Sequence[int] = (2, 5),
) -> Dict[str, float]:
    """Mean Recall@k (percent) over all queries for each k in ks."""
    assert len(retrieved_titles) == len(gold_titles)
    n = len(gold_titles)
    out: Dict[str, float] = {}
    for k in ks:
        if n == 0:
            out[f"recall@{k}"] = 0.0
        else:
            out[f"recall@{k}"] = (
                100.0 * sum(recall_at_k(r, g, k) for r, g in zip(retrieved_titles, gold_titles)) / n
            )
    return out
