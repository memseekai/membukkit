"""Export a shared-corpus multi-hop split as markdown, for other retrievers.

    python -m benchmarks.multihop.export_corpus --dataset musique --out /tmp/musique_corpus

Writes one markdown file per passage into a single directory, plus a
``queries.json`` in QMD's fixture format so ``qmd bench`` can be pointed
straight at it:

    qmd init && qmd collection add <out> --name musique && qmd embed
    qmd bench <out>/queries.json --collection musique --json

Unlike the HotpotQA distractor exporter, there is **one** directory here, not
one per question: the whole point of these splits is that every question is
answered against the same corpus.

Gold labels are exported as filenames, which is what QMD reports and matches on,
so both systems are scored against the same targets. Supporting-fact labels
never enter the passage files themselves.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

from benchmarks.common.paths import unique_filename
from benchmarks.multihop import dataset as ds


def export(out_dir: pathlib.Path, dataset: str = "musique",
           limit: int | None = None, seed: int = 42,
           top_k: int = 10) -> Dict:
    split = ds.load_split(dataset, limit=limit, seed=seed)
    docs_dir = out_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Filenames must be unique or gold matching silently breaks.
    used: Dict[str, int] = {}
    file_of_doc: Dict[str, str] = {}
    for p in split.passages:
        name = unique_filename(p.title, used)
        (docs_dir / name).write_text(p.text)
        file_of_doc[p.doc_id] = name

    queries = []
    for q in split.questions:
        # Gold is the specific supporting passages. Expanding a gold *title* to
        # every passage sharing it inflates MuSiQue gold sets to as many as 41
        # files and caps recall near 0.24, which measures the wrong thing.
        expected = [file_of_doc[d] for d in ds.gold_doc_ids(q, split.passages)
                    if d in file_of_doc]
        if not expected:
            continue
        queries.append({
            "id": q.qid,
            "query": q.question,
            "type": q.hop_type,
            "description": f"{split.name} {q.hop_type} question",
            "expected_files": expected,
            "expected_in_top_k": top_k,
        })

    (out_dir / "queries.json").write_text(json.dumps({
        "description": f"{split.name} shared-corpus multi-hop retrieval",
        "version": 1,
        "collection": split.name,
        "queries": queries,
    }, indent=2) + "\n")

    index = {
        "dataset": split.name,
        "source": "official HippoRAG release split",
        "n_questions": len(queries),
        "n_passages": len(split.passages),
        "limit": limit,
        "seed": seed,
        "qa_sha256": split.qa_sha256,
        "corpus_sha256": split.corpus_sha256,
        "docs_dir": str(docs_dir),
        "note": ("Single shared corpus. Index it once and answer every query "
                 "against all of it; splitting it per question changes the task."),
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="musique", choices=sorted(ds.ALIASES))
    ap.add_argument("--limit", type=int, default=0, help="0 = the full split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    index = export(args.out, dataset=args.dataset, limit=args.limit or None,
                   seed=args.seed, top_k=args.top_k)
    print(f"exported {index['n_passages']} passages, {index['n_questions']} queries")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
