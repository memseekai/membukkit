"""Official HippoRAG multi-hop splits: MuSiQue, 2WikiMultiHopQA, HotpotQA.

These are *shared-corpus* benchmarks and that is the point. Every question is
answered against the whole corpus (MuSiQue: 1,000 questions over 11,656
passages), not against its own 10-paragraph candidate set. Candidate-set
retrieval saturates quickly; shared-corpus retrieval does not, so this is where
retrieval strategies actually separate.

Source: the release files in the official HippoRAG repository, so numbers stay
comparable with published results that use the same splits.

Guardrail: split sizes are asserted against the official counts and the loader
fails loudly on a mismatch. A silently-substituted variant would invalidate any
comparison drawn from it.

Fairness rules enforced here:

- Passage text is ``"{title}\\n{text}"``, matching how HippoRAG indexes a
  passage. Nothing is rewritten, reordered, or summarised.
- The answer string is never indexed.
- Supporting-title labels never enter passage text or searchable metadata; they
  live only in the evaluator-side ``gold_titles`` field.
- The question is never mixed into ingestion.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import random
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_RAW_BASE = (
    "https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/main/reproduce/dataset"
)
DEFAULT_CACHE = pathlib.Path.home() / ".cache" / "membukkit" / "multihop"
_DOWNLOAD_TIMEOUT_S = 600

_FILES = {
    "musique": ("musique.json", "musique_corpus.json"),
    "2wikimultihopqa": ("2wikimultihopqa.json", "2wikimultihopqa_corpus.json"),
    "hotpotqa": ("hotpotqa.json", "hotpotqa_corpus.json"),
}

ALIASES = {
    "musique": "musique",
    "2wiki": "2wikimultihopqa",
    "2wikimultihopqa": "2wikimultihopqa",
    "hotpot": "hotpotqa",
    "hotpotqa": "hotpotqa",
}

# (questions, corpus passages) the official splits MUST have.
_EXPECTED: Dict[str, Tuple[int, int]] = {
    "musique": (1000, 11656),
    "2wikimultihopqa": (1000, 6119),
    "hotpotqa": (1000, 9811),
}


@dataclass
class Passage:
    doc_id: str
    title: str
    text: str


@dataclass
class Question:
    qid: str
    question: str
    gold_titles: List[str] = field(default_factory=list)  # evaluator-side only
    hop_type: str = "multihop"
    gold_doc_ids: List[str] = field(default_factory=list)  # evaluator-side only

    @property
    def n_gold(self) -> int:
        return len(self.gold_doc_ids or self.gold_titles)


@dataclass
class Split:
    name: str
    questions: List[Question]
    passages: List[Passage]
    qa_sha256: str
    corpus_sha256: str


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_file(name: str, cache_dir: pathlib.Path) -> pathlib.Path:
    local = cache_dir / name
    if local.exists():
        return local
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"{_RAW_BASE}/{name}"
    logger.info("fetching official split %s", url)
    tmp = local.with_suffix(local.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_S) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(local)  # atomic: a half-written split must never look cached
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not fetch {name!r} from the official HippoRAG repo ({url}). "
            f"Substituting a different variant would invalidate the comparison, "
            f"so this is fatal. Error: {e}"
        ) from e
    return local


def passage_text(title: str, text: str) -> str:
    """HippoRAG indexes a passage as 'title\\ntext'. Match that exactly."""
    return f"{title}\n{text}"


def _gold_paragraphs(record: Dict) -> List[Tuple[str, Optional[str]]]:
    """Gold as ``(title, paragraph_text)``; text is ``None`` when unavailable.

    MuSiQue ships the supporting paragraph's own text, which matters because its
    corpus repeats titles (647 of them, "New York City" 28 times). Matching gold
    on title alone would mark every passage under that title as expected, so a
    question with 2 gold paragraphs would demand up to 41 files and cap recall
    near 0.24. Matching on (title, text) resolves all 2,648 MuSiQue gold
    paragraphs to exactly one passage each.

    2Wiki and HotpotQA give only ``supporting_facts`` titles, but their corpora
    have no duplicate titles, so title matching is already exact there.
    """
    paragraphs = record.get("paragraphs") or []
    supporting = [(p["title"], p.get("paragraph_text"))
                  for p in paragraphs if p.get("is_supporting")]
    if supporting:
        return supporting
    return [(sf[0], None) for sf in (record.get("supporting_facts") or []) if sf]


def _gold_titles(record: Dict) -> List[str]:
    paragraphs = record.get("paragraphs") or []
    supporting = {p["title"] for p in paragraphs if p.get("is_supporting")}
    if supporting:
        return sorted(supporting)
    # 2Wiki/HotpotQA style: supporting_facts is a list of [title, sent_id].
    return sorted({sf[0] for sf in (record.get("supporting_facts") or []) if sf})


def _qid(record: Dict, index: int) -> str:
    """Stable question id.

    The splits disagree on the field name: MuSiQue uses ``id``, 2Wiki and
    HotpotQA use ``_id``. Reading only one of them silently yields the string
    ``"None"`` for every question in the other, which collapses a whole run into
    a single result. Hence the explicit fallback chain and the duplicate check
    in :func:`load_split`.
    """
    for key in ("id", "_id", "qid"):
        value = record.get(key)
        if value not in (None, "", "None"):
            return str(value)
    return f"q{index}"


def _hop_type(record: Dict, qid: str) -> str:
    """Question category: the split's own label when it has one."""
    declared = record.get("type")
    if declared:
        return str(declared)
    return f"{qid.split('hop')[0]}hop" if "hop" in qid else "multihop"


def load_split(
    dataset: str,
    *,
    limit: Optional[int] = None,
    seed: int = 42,
    cache_dir: Optional[pathlib.Path] = None,
    strict: bool = True,
) -> Split:
    """Load an official split, optionally sampling ``limit`` questions.

    Sampling is deterministic in ``(dataset, limit, seed)`` and runs over the
    id-sorted question list, so the order of the source file cannot change which
    questions are selected. The corpus is never sampled: every question is
    always scored against the full corpus.
    """
    key = ALIASES.get(dataset.lower())
    if key is None:
        raise ValueError(f"unknown dataset {dataset!r}; use one of {sorted(ALIASES)}")

    cache = pathlib.Path(cache_dir) if cache_dir else DEFAULT_CACHE
    qa_name, corpus_name = _FILES[key]
    qa_path = _ensure_file(qa_name, cache)
    corpus_path = _ensure_file(corpus_name, cache)

    qa = json.loads(qa_path.read_text())
    corpus = json.loads(corpus_path.read_text())
    exp_q, exp_c = _EXPECTED[key]
    if strict and (len(qa), len(corpus)) != (exp_q, exp_c):
        raise AssertionError(
            f"size mismatch for {key}: got {len(qa)} questions / {len(corpus)} "
            f"passages, expected {exp_q}/{exp_c}. The official split changed or "
            f"the wrong file was fetched; refusing to proceed."
        )

    passages = [
        Passage(doc_id=f"{i}:{c['title']}", title=c["title"],
                text=passage_text(c["title"], c["text"]))
        for i, c in enumerate(corpus)
    ]

    # Resolve gold to concrete passages. Exact (title, text) where the split
    # provides paragraph text, else every passage under that title.
    by_key = {(p.title, p.text): p.doc_id for p in passages}
    by_title: Dict[str, List[str]] = {}
    for p in passages:
        by_title.setdefault(p.title, []).append(p.doc_id)

    questions = []
    for i, ex in enumerate(qa):
        qid = _qid(ex, i)
        gold_ids: List[str] = []
        for title, text in _gold_paragraphs(ex):
            exact = by_key.get((title, passage_text(title, text))) if text else None
            if exact:
                gold_ids.append(exact)
            else:
                gold_ids.extend(by_title.get(title, []))
        questions.append(Question(qid=qid, question=ex.get("question", ""),
                                  gold_titles=_gold_titles(ex),
                                  hop_type=_hop_type(ex, qid),
                                  gold_doc_ids=list(dict.fromkeys(gold_ids))))
    questions = [q for q in questions if q.gold_doc_ids and q.question]

    # Duplicate ids silently collapse a run: downstream reports key results by
    # question id, so N questions sharing one id become one result carrying the
    # last question's ranking. Fail loudly instead.
    seen = {}
    for q in questions:
        if q.qid in seen:
            raise AssertionError(
                f"duplicate question id {q.qid!r} in {key}: ids must be unique or "
                f"per-question results collapse into one. First seen as "
                f"{seen[q.qid]!r}, again as {q.question!r}."
            )
        seen[q.qid] = q.question
    questions.sort(key=lambda q: q.qid)
    if limit is not None and limit < len(questions):
        questions = sorted(random.Random(seed).sample(questions, limit),
                           key=lambda q: q.qid)

    logger.info("[%s] %d questions over %d passages", key, len(questions), len(passages))
    return Split(name=key, questions=questions, passages=passages,
                 qa_sha256=_sha256(qa_path), corpus_sha256=_sha256(corpus_path))


def gold_doc_ids(question: Question, passages: List[Passage]) -> List[str]:
    """Evaluator-side only: the resolved supporting passages for ``question``."""
    if question.gold_doc_ids:
        return list(question.gold_doc_ids)
    wanted = set(question.gold_titles)
    return [p.doc_id for p in passages if p.title in wanted]


def title_of(doc_id: str) -> str:
    """Recover the title from a doc_id built by :func:`load_split`."""
    return doc_id.split(":", 1)[1] if ":" in doc_id else doc_id
