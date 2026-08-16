"""Export the HotpotQA candidate corpus as markdown, for other retrievers.

    python -m benchmarks.hotpotqa.export_corpus --limit 100 --seed 42 --out /tmp/hotpot_corpus

Writes one directory per question, each containing that question's candidate
paragraphs as markdown files, plus a ``queries.json`` in QMD's fixture format
so ``qmd bench`` can be pointed straight at it.

Per-question directories exist because the distractor setting is candidate-set
retrieval: a question must be scored against its own 10 paragraphs only. Any
retriever run over a merged corpus would be solving a different, much harder
task and the numbers would not be comparable.

Filenames are the slugified title, which is what QMD reports as ``filepath``
and what its scorer matches on, so the same gold labels work for both systems.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Dict, List

from benchmarks.hotpotqa import dataset as ds


def slugify(title: str) -> str:
    """Filesystem-safe, collision-resistant filename for a document title."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", title).strip("-").lower()
    return (slug or "untitled")[:120]


def export(
    out_dir: pathlib.Path,
    split: str = "validation",
    limit: int | None = 100,
    seed: int = 42,
) -> Dict:
    questions = ds.load_questions(split=split, limit=limit, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict] = []
    for q in questions:
        qdir = out_dir / q.qid
        qdir.mkdir(parents=True, exist_ok=True)

        # Slugs must stay unique inside a question or gold matching breaks.
        used: Dict[str, int] = {}
        title_to_file: Dict[str, str] = {}
        for doc in q.docs:
            base = slugify(doc["title"])
            n = used.get(base, 0)
            used[base] = n + 1
            name = f"{base}.md" if n == 0 else f"{base}-{n}.md"
            (qdir / name).write_text(doc["text"])
            title_to_file[doc["title"]] = name

        manifest.append(
            {
                "qid": q.qid,
                "dir": str(qdir),
                "query": q.question,
                "type": q.qtype,
                "expected_files": [
                    title_to_file[t] for t in q.gold_titles if t in title_to_file
                ],
                "n_candidates": len(q.docs),
            }
        )

        # QMD fixture format, one per question, so `qmd bench` can run directly.
        (qdir / "queries.json").write_text(
            json.dumps(
                {
                    "description": f"HotpotQA distractor question {q.qid}",
                    "version": 1,
                    "collection": q.qid,
                    "queries": [
                        {
                            "id": q.qid,
                            "query": q.question,
                            "type": q.qtype,
                            "description": f"HotpotQA {q.qtype} question",
                            "expected_files": [
                                title_to_file[t] for t in q.gold_titles if t in title_to_file
                            ],
                            "expected_in_top_k": 10,
                        }
                    ],
                },
                indent=2,
            )
            + "\n"
        )

    index = {
        "dataset": "hotpotqa-distractor",
        "split": split,
        "limit": limit,
        "seed": seed,
        "n_questions": len(questions),
        "note": (
            "One directory per question. Each must be indexed and searched in "
            "isolation; merging them changes the task."
        ),
        "questions": manifest,
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="validation", choices=["validation", "train"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    limit = None if args.limit in (0, -1) else args.limit
    index = export(args.out, split=args.split, limit=limit, seed=args.seed)
    total_docs = sum(q["n_candidates"] for q in index["questions"])
    print(f"exported {index['n_questions']} questions, {total_docs} documents")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
