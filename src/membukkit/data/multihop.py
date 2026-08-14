"""Multi-hop RAG dataset adapters: MuSiQue, 2WikiMultiHopQA, HotpotQA.

These are the HippoRAG-released dev splits (1000 questions each) with the
candidate-passage retrieval corpus (supporting + distractor passages tied to those
questions — NOT full Wikipedia), pulled byte-identically from the official repo
OSU-NLP-Group/HippoRAG (reproduce/dataset/*.json). This is the exact protocol the
anchor paper (MultiCube-RAG, arXiv 2602.15898, Table 1) and HippoRAG/HippoRAG 2 use.

Design: each question becomes a `MultiHopInstance` that SUBCLASSES the project's
QA-eval contract (`data.instance.LongMemEvalInstance`) so it plugs into the
existing harness unchanged — `get_facts()` returns the shared corpus, `get_queries()`
returns the question. The single retrieval corpus is shared by reference across all
1000 instances (it is one pool per dataset, not per-question). The dedicated runner
reads `MultiHopDataset.corpus` directly to embed the corpus once.

Gold signals:
  - gold_answers : [answer] (+ answer_aliases for MuSiQue) -> EM/F1 (SQuAD scorer)
  - gold_titles  : supporting-passage titles -> passage Recall@2 / Recall@5

ASSERTS sizes and logs a split hash; fails loudly on any mismatch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from membukkit.data.base import FactInput, QueryInput
from membukkit.data.instance import LongMemEvalDataset, LongMemEvalInstance

logger = logging.getLogger(__name__)

_EPOCH = datetime(2024, 1, 1)

_FILES = {
    "musique": ("musique.json", "musique_corpus.json"),
    "2wikimultihopqa": ("2wikimultihopqa.json", "2wikimultihopqa_corpus.json"),
    "hotpotqa": ("hotpotqa.json", "hotpotqa_corpus.json"),
}

_ALIASES = {
    "musique": "musique",
    "2wiki": "2wikimultihopqa",
    "2wikimultihopqa": "2wikimultihopqa",
    "hotpot": "hotpotqa",
    "hotpotqa": "hotpotqa",
}

_EXPECTED: Dict[str, Tuple[int, int]] = {
    "musique": (1000, 11656),
    "2wikimultihopqa": (1000, 6119),
    "hotpotqa": (1000, 9811),
}

_RAW_BASE = "https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/main/reproduce/dataset"
_DEFAULT_DIR = Path(__file__).resolve().parent / "multihop_splits"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_file(name: str, data_dir: Path) -> Path:
    local = data_dir / name
    if local.exists():
        return local
    data_dir.mkdir(parents=True, exist_ok=True)
    url = f"{_RAW_BASE}/{name}"
    logger.info(f"Fetching official split from {url}")
    try:
        urllib.request.urlretrieve(url, local)
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch '{name}' from the official HippoRAG repo ({url}). "
            f"Aborting. Error: {e}"
        )
    return local


def _passage_text(title: str, text: str) -> str:
    """HippoRAG indexes a passage as 'title\\ntext'. Match that exactly."""
    return f"{title}\n{text}"


class MultiHopInstance(LongMemEvalInstance):
    """One multi-hop question over a SHARED candidate-passage corpus."""

    def __init__(self, item: Dict, shared_facts: List[FactInput]):
        super().__init__(item)
        self._facts = shared_facts

    @property
    def ability(self) -> str:
        return self._item.get("hop_type", self._item.get("question_type", "multihop"))

    @property
    def gold_answers(self) -> List[str]:
        return self._item.get("gold_answers", [])

    @property
    def gold_titles(self) -> List[str]:
        return self._item.get("gold_titles", [])

    def get_queries(self) -> List[QueryInput]:
        if self._queries is None:
            self._queries = [QueryInput(
                text=self._item.get("question", ""),
                query_type=self.ability,
                ground_truth=self._item.get("answer", ""),
                evidence=self.gold_titles,
                category=None,
            )]
        return self._queries


class MultiHopDataset(LongMemEvalDataset):
    """A multi-hop benchmark: 1000 questions + one shared candidate-passage corpus."""

    instances: List[MultiHopInstance]

    def __init__(
        self,
        dataset_name: str,
        instances: List[MultiHopInstance],
        corpus: List[Dict],
        split_hash: str,
        qa_hash: str,
    ):
        super().__init__(instances)
        self.dataset_name = dataset_name
        self.corpus = corpus
        self.passage_titles = [c["title"] for c in corpus]
        self.passages_text = [_passage_text(c["title"], c["text"]) for c in corpus]
        self.split_hash = split_hash
        self.qa_hash = qa_hash

    def provenance(self) -> Dict:
        return {
            "dataset": self.dataset_name,
            "n_questions": len(self.instances),
            "n_passages": len(self.corpus),
            "qa_sha256": self.qa_hash,
            "corpus_sha256": self.split_hash,
        }


def _extract_2wiki_hotpot(qa: List[Dict], dataset: str) -> List[Dict]:
    out = []
    for ex in qa:
        qid = ex.get("_id") or ex.get("id")
        gold_titles = sorted({t for t, _ in ex.get("supporting_facts", [])})
        out.append({
            "question_id": str(qid),
            "question": ex.get("question", ""),
            "answer": str(ex.get("answer", "")),
            "gold_answers": [str(ex.get("answer", ""))],
            "gold_titles": gold_titles,
            "hop_type": ex.get("type", "multihop"),
        })
    return out


def _extract_musique(qa: List[Dict]) -> List[Dict]:
    out = []
    for ex in qa:
        qid = ex.get("id")
        gold_titles = sorted({
            p["title"] for p in ex.get("paragraphs", []) if p.get("is_supporting")
        })
        ans = str(ex.get("answer", ""))
        aliases = [str(a) for a in ex.get("answer_aliases", []) if a]
        hop = qid.split("hop")[0] + "hop" if isinstance(qid, str) and "hop" in qid else "multihop"
        out.append({
            "question_id": str(qid),
            "question": ex.get("question", ""),
            "answer": ans,
            "gold_answers": [ans] + aliases,
            "gold_titles": gold_titles,
            "hop_type": hop,
        })
    return out


def load_multihop(
    dataset: str,
    data_dir: Optional[str] = None,
    max_questions: Optional[int] = None,
    strict: bool = True,
) -> MultiHopDataset:
    """Load a multi-hop benchmark from the official HippoRAG release split.

    Args:
        dataset: 'musique' | '2wiki'/'2wikimultihopqa' | 'hotpot'/'hotpotqa'.
        data_dir: where the official *.json live / are cached.
        max_questions: cap (smoke runs ONLY; never report a capped run as a result).
        strict: if True (default), ASSERT exact official sizes and fail loudly.
    """
    key = _ALIASES.get(dataset.lower())
    if key is None:
        raise ValueError(f"Unknown dataset '{dataset}'. Use one of {sorted(_ALIASES)}.")

    ddir = Path(data_dir) if data_dir else _DEFAULT_DIR
    qa_name, corpus_name = _FILES[key]
    qa_path = _ensure_file(qa_name, ddir)
    corpus_path = _ensure_file(corpus_name, ddir)

    qa = json.loads(qa_path.read_text())
    corpus = json.loads(corpus_path.read_text())
    qa_hash = _sha256(qa_path)
    corpus_hash = _sha256(corpus_path)

    exp_q, exp_c = _EXPECTED[key]
    n_q, n_c = len(qa), len(corpus)
    logger.info(
        f"[{key}] loaded {n_q} questions / {n_c} passages "
        f"(expected {exp_q}/{exp_c}); qa_sha256={qa_hash[:12]} corpus_sha256={corpus_hash[:12]}"
    )
    if strict and (n_q != exp_q or n_c != exp_c):
        raise AssertionError(
            f"SIZE MISMATCH for {key}: got {n_q} questions / {n_c} passages, "
            f"expected {exp_q}/{exp_c}. The official split changed or the wrong file "
            f"was fetched — refusing to proceed. "
            f"qa={qa_path} corpus={corpus_path}"
        )

    shared_facts = [
        FactInput(text=_passage_text(c["title"], c["text"]), timestamp=_EPOCH,
                  tag="NEW_OBS", source_session=c["title"])
        for c in corpus
    ]

    items = (_extract_musique(qa) if key == "musique"
             else _extract_2wiki_hotpot(qa, key))
    if max_questions is not None:
        items = items[:max_questions]

    instances = [MultiHopInstance(it, shared_facts) for it in items]
    logger.info(f"[{key}] built {len(instances)} instances over shared corpus of {n_c} passages")

    return MultiHopDataset(key, instances, corpus, corpus_hash, qa_hash)
