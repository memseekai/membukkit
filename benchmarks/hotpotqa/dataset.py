"""HotpotQA distractor split loader and per-question corpus construction.

The distractor setting gives each question its own candidate set: 10 paragraphs,
of which 2 are the gold supporting documents and 8 are distractors chosen to
look plausible. That makes this *candidate-set* retrieval, not retrieval over
all of Wikipedia, and results must never be read as the latter.

Source: the Hugging Face parquet conversion of ``hotpotqa/hotpot_qa`` (config
``distractor``). The original CMU host (curtis.ml.cmu.edu) is used by the
official README but is frequently unreachable, so it is not relied on here.

Fairness rules enforced in this module:

- Document text is the HotpotQA paragraph verbatim under a ``# {title}``
  heading. Sentences are joined as stored; nothing is rewritten or reordered.
- The answer string is never indexed.
- Supporting-fact labels never enter document text or searchable metadata; they
  exist only in the evaluator-side ``gold_titles`` field.
- The question is never mixed into ingestion.
"""

from __future__ import annotations

import hashlib
import logging
import pathlib
import random
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_PARQUET = (
    "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/distractor/"
    "{split}/0.parquet"
)
_SPLITS = {"validation", "train"}
DEFAULT_CACHE = pathlib.Path.home() / ".cache" / "membukkit" / "hotpotqa"
_DOWNLOAD_TIMEOUT_S = 300


@dataclass
class HotpotQuestion:
    """One question with its own candidate corpus and evaluator-side labels."""

    qid: str
    question: str
    qtype: str  # "bridge" | "comparison"
    level: str  # "easy" | "medium" | "hard"
    docs: List[Dict[str, str]] = field(default_factory=list)  # {doc_id, title, text}
    gold_titles: List[str] = field(default_factory=list)  # evaluator-side only

    @property
    def n_gold(self) -> int:
        return len(self.gold_titles)


def _ensure_file(split: str, cache_dir: pathlib.Path) -> pathlib.Path:
    if split not in _SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(_SPLITS)}")
    local = cache_dir / f"hotpot_{split}_distractor.parquet"
    if local.exists():
        return local
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = _PARQUET.format(split=split)
    print(f"downloading HotpotQA {split} (distractor) parquet, one time...")
    tmp = local.with_suffix(".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "membukkit-bench"})
        with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT_S) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(local)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"could not download HotpotQA {split} from {url}: {e}") from e
    return local


def _read_records(path: pathlib.Path) -> List[Dict]:
    """Read the parquet into plain dicts. Needs the ``bench`` extra (pyarrow)."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - dependency guard
        raise ImportError(
            "reading the HotpotQA parquet needs pyarrow: "
            'pip install "membukkit[bench]"  (or: uv run --with pyarrow ...)'
        ) from e
    return pq.read_table(path).to_pylist()


def stable_doc_id(qid: str, title: str) -> str:
    """Deterministic per-question document id.

    Scoped by question because the same Wikipedia title appears in several
    questions' candidate sets, and each question is indexed in isolation.
    """
    digest = hashlib.sha1(f"{qid}\x1f{title}".encode()).hexdigest()[:10]
    return f"{title}#{digest}"


def render_document(title: str, sentences: Sequence[str]) -> str:
    """Deterministic markdown for one paragraph. Text is left as published."""
    body = "".join(sentences).strip()
    return f"# {title}\n\n{body}\n"


def question_from_record(rec: Dict) -> HotpotQuestion:
    """Build one question from a HF ``hotpot_qa`` row.

    The HF schema is columnar: ``supporting_facts`` is ``{title: [...],
    sent_id: [...]}`` and ``context`` is ``{title: [...], sentences: [[...]]}``.
    """
    sf = rec.get("supporting_facts") or {}
    gold_titles = sorted(set(sf.get("title") or []))

    ctx = rec.get("context") or {}
    titles = list(ctx.get("title") or [])
    sentence_lists = list(ctx.get("sentences") or [])

    qid = rec.get("id") or rec.get("_id") or ""
    docs = []
    for title, sentences in zip(titles, sentence_lists):
        docs.append(
            {
                "doc_id": stable_doc_id(qid, title),
                "title": title,
                "text": render_document(title, list(sentences)),
            }
        )
    return HotpotQuestion(
        qid=qid,
        question=rec.get("question", ""),
        qtype=rec.get("type", "unknown"),
        level=rec.get("level", "unknown"),
        docs=docs,
        gold_titles=gold_titles,
    )


def questions_from_records(
    records: Sequence[Dict],
    limit: Optional[int] = None,
    seed: int = 42,
) -> List[HotpotQuestion]:
    """Deterministically sample and convert records.

    Sampling is seeded and applied to the id-sorted list, so the same
    ``(records, limit, seed)`` always yields the same questions regardless of
    the order the source file happens to be in.
    """
    items = sorted(records, key=lambda r: r.get("id") or r.get("_id") or "")
    if limit is not None and limit < len(items):
        items = random.Random(seed).sample(items, limit)
        items.sort(key=lambda r: r.get("id") or r.get("_id") or "")
    return [question_from_record(r) for r in items]


def load_questions(
    split: str = "validation",
    limit: Optional[int] = None,
    seed: int = 42,
    cache_dir: Optional[pathlib.Path] = None,
) -> List[HotpotQuestion]:
    """Download (once), then deterministically sample the split."""
    path = _ensure_file(split, cache_dir or DEFAULT_CACHE)
    return questions_from_records(_read_records(path), limit=limit, seed=seed)


def gold_doc_ids(q: HotpotQuestion) -> List[str]:
    """Evaluator-side mapping from gold titles to this question's doc ids."""
    by_title = {d["title"]: d["doc_id"] for d in q.docs}
    return [by_title[t] for t in q.gold_titles if t in by_title]
