"""Run QMD over the exported HotpotQA corpus, scored identically to MemBukkit.

    python -m benchmarks.hotpotqa.export_corpus --limit 100 --seed 42 --out /tmp/hotpot_corpus
    python -m benchmarks.hotpotqa.run_qmd --corpus /tmp/hotpot_corpus --qmd-bin /path/to/qmd/bin/qmd

Each question gets a **fresh QMD index** containing only its own candidate
paragraphs, matching the isolation the MemBukkit runner uses. Sharing one index
across questions would turn candidate-set retrieval into something else
entirely and make the two sets of numbers incomparable.

QMD's bench emits ``top_files`` (its ranked document list) per backend. We take
that list, strip the ``qmd://<collection>/`` prefix, and feed it through the
*same* metric functions the MemBukkit runner uses, so any-support and
all-support recall mean exactly the same thing on both sides.

QMD's four backends are reported separately:
  bm25    lexical only
  vector  dense only
  hybrid  BM25 + vector fused, no reranker
  full    hybrid plus cross-encoder reranking (its recommended path)
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import time
from typing import Dict, List

from benchmarks.common import metrics, qmd_compat

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
KS = (1, 3, 5, 10)
BACKENDS = ("bm25", "vector", "hybrid", "full")


def _run(cmd: List[str], env: Dict[str, str], cwd: pathlib.Path, timeout: int = 600):
    return subprocess.run(
        cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )


def bench_one_question(
    qmd_bin: pathlib.Path,
    qdir: pathlib.Path,
    workdir: pathlib.Path,
    shared_home: pathlib.Path,
    collection: str,
) -> Dict | None:
    """Index one question's candidate set in a fresh index, then bench it.

    Two separate isolation concerns:

    - **Documents** live in a project-local ``.qmd/index.sqlite`` created by
      ``qmd init`` in ``workdir``, which is fresh per question. That is what
      keeps one question's paragraphs out of another's ranking.
    - **Collection names** live in a registry under ``$HOME``. QMD reads
      ``HOME`` (see its ``qmdHomedir``), not ``QMD_HOME``, so ``HOME`` is
      pointed at one shared benchmark directory: it keeps the run out of the
      user's real ``~/.qmd`` while still reusing the ~1GB of downloaded GGUF
      models across questions. Because that registry is shared, each question
      needs a unique ``--name`` or the second one collides.
    """
    env = {**os.environ, "HOME": str(shared_home)}
    workdir.mkdir(parents=True, exist_ok=True)

    for cmd in (
        ["node", str(qmd_bin), "init"],
        ["node", str(qmd_bin), "collection", "add", str(qdir), "--name", collection],
        ["node", str(qmd_bin), "embed"],
    ):
        r = _run(cmd, env, workdir)
        if r.returncode != 0:
            return {"error": f"{cmd[2]} failed: {(r.stderr or r.stdout)[-300:]}"}

    r = _run(
        ["node", str(qmd_bin), "bench", str(qdir / "queries.json"),
         "--collection", collection, "--json"],
        env,
        workdir,
    )
    if r.returncode != 0:
        return {"error": f"bench failed: {(r.stderr or r.stdout)[-300:]}"}
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"error": f"unparseable bench output: {r.stdout[:300]}"}


def run(
    corpus: pathlib.Path,
    qmd_bin: pathlib.Path,
    limit: int | None = None,
    progress_every: int = 10,
) -> Dict:
    index = json.loads((corpus / "index.json").read_text())
    questions = index["questions"][: limit or len(index["questions"])]
    print(f"benchmarking QMD over {len(questions)} HotpotQA questions", flush=True)

    per_query: List[Dict] = []
    t0 = time.perf_counter()
    tmp_root = pathlib.Path(tempfile.mkdtemp(prefix="qmdbench-"))
    # One shared HOME: isolates the run from the user's ~/.qmd, but keeps the
    # downloaded models cached across questions instead of re-fetching ~1GB.
    shared_home = tmp_root / "home"
    shared_home.mkdir(parents=True, exist_ok=True)
    for sub in ("models",):
        src = pathlib.Path.home() / ".cache" / "qmd" / sub
        if src.exists():
            (shared_home / ".cache" / "qmd").mkdir(parents=True, exist_ok=True)
            try:
                (shared_home / ".cache" / "qmd" / sub).symlink_to(src)
            except OSError:
                pass

    try:
        for i, q in enumerate(questions, start=1):
            work = tmp_root / "q" / q["qid"]
            out = bench_one_question(
                qmd_bin, pathlib.Path(q["dir"]), work, shared_home, f"bench{i}"
            )
            shutil.rmtree(work, ignore_errors=True)  # keep disk bounded

            if not out or "error" in (out or {}):
                per_query.append({"qid": q["qid"], "type": q["type"],
                                  "error": (out or {}).get("error", "unknown")})
                continue

            gold = q["expected_files"]
            row = {"qid": q["qid"], "type": q["type"], "n_gold": len(gold),
                   "expected_docs": gold, "backends": {}}
            for name in BACKENDS:
                b = out["results"][0]["backends"].get(name)
                if not b:
                    continue
                # Strip qmd://<collection>/ so ids match the exported filenames.
                ranked = [qmd_compat.normalize_path(f) for f in b.get("top_files", [])]
                row["backends"][name] = {
                    "retrieved_docs": ranked[:10],
                    "latency_ms": b.get("latency_ms"),
                    "first_relevant_rank": metrics.first_relevant_rank(ranked, gold),
                    "metrics": {
                        **{f"recall@{k}": metrics.recall_at_k(ranked, gold, k) for k in KS},
                        **{f"any@{k}": metrics.any_support_at_k(ranked, gold, k) for k in KS},
                        **{f"all@{k}": metrics.all_support_at_k(ranked, gold, k) for k in KS},
                        "mrr": metrics.reciprocal_rank(ranked, gold),
                        "ndcg@10": metrics.ndcg_at_k(ranked, gold, 10),
                    },
                }
            per_query.append(row)

            if progress_every and i % progress_every == 0:
                el = time.perf_counter() - t0
                print(f"  {i}/{len(questions)}  ({el:.0f}s, {el / i:.1f}s/question)", flush=True)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    ok = [r for r in per_query if "backends" in r]
    summary: Dict[str, Dict] = {}
    for name in BACKENDS:
        rows = [
            {"metrics": r["backends"][name]["metrics"],
             "latency_ms": r["backends"][name]["latency_ms"]}
            for r in ok if name in r["backends"]
        ]
        if not rows:
            continue
        summary[name] = metrics.aggregate(rows, ks=KS)
        for subset in ("bridge", "comparison"):
            srows = [
                {"metrics": r["backends"][name]["metrics"],
                 "latency_ms": r["backends"][name]["latency_ms"]}
                for r in ok if r["type"] == subset and name in r["backends"]
            ]
            if srows:
                summary[name][subset] = metrics.aggregate(srows, ks=KS)

    return {
        "benchmark": "hotpotqa-distractor-qmd",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": "qmd",
        "dataset": {k: index[k] for k in ("split", "limit", "seed") if k in index},
        "n_questions": len(questions),
        "n_scored": len(ok),
        "n_failed": len(per_query) - len(ok),
        "isolation": "fresh QMD index per question (candidate-set retrieval)",
        "wall_clock_s": time.perf_counter() - t0,
        "summary": summary,
        "per_query": per_query,
    }


def print_table(report: Dict) -> None:
    print(f"\nQMD on HotpotQA distractor, {report['n_scored']}/{report['n_questions']} scored")
    if report["n_failed"]:
        print(f"  ({report['n_failed']} failed)")
    print()
    print(f"{'Backend':<10}{'Any@1':>8}{'Any@3':>8}{'All@3':>8}{'All@5':>8}"
          f"{'All@10':>8}{'MRR':>8}{'Latency':>10}")
    print("-" * 68)
    for name in BACKENDS:
        s = report["summary"].get(name)
        if not s:
            continue
        print(f"{name:<10}{s.get('any@1', 0):>8.3f}{s.get('any@3', 0):>8.3f}"
              f"{s.get('all@3', 0):>8.3f}{s.get('all@5', 0):>8.3f}{s.get('all@10', 0):>8.3f}"
              f"{s.get('mrr', 0):>8.3f}{s.get('latency_ms_mean', 0):>8.0f}ms")

    print(f"\n{'Backend':<10}{'Subset':<13}{'N':>5}{'Any@5':>9}{'All@5':>9}{'MRR':>9}")
    print("-" * 55)
    for name in BACKENDS:
        s = report["summary"].get(name) or {}
        for subset in ("bridge", "comparison"):
            ss = s.get(subset)
            if ss:
                print(f"{name:<10}{subset:<13}{int(ss.get('n', 0)):>5}{ss.get('any@5', 0):>9.3f}"
                      f"{ss.get('all@5', 0):>9.3f}{ss.get('mrr', 0):>9.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=pathlib.Path, required=True,
                    help="directory written by benchmarks.hotpotqa.export_corpus")
    ap.add_argument("--qmd-bin", type=pathlib.Path, required=True,
                    help="path to qmd's bin/qmd")
    ap.add_argument("--limit", type=int, default=0, help="cap questions (0 = all exported)")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    report = run(args.corpus, args.qmd_bin, limit=args.limit or None)
    print_table(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = args.out or RESULTS_DIR / f"hotpotqa_qmd_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
