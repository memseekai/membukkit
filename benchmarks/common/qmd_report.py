"""Emit and compare reports in QMD's ``qmd bench --json`` format.

The point is falsifiability. Running MemBukkit through QMD's own fixture format
and emitting QMD's own report schema means a comparison needs no trust in our
harness: index the same corpus, run both binaries, diff two JSON files.

    python -m benchmarks.multihop.export_corpus --dataset musique --out /tmp/musique
    python -m benchmarks.multihop.run_fixture --corpus /tmp/musique --out mb.json
    qmd bench /tmp/musique/queries.json --collection musique --json > qmd.json
    python -m benchmarks.common.qmd_report mb.json qmd.json

Schema, matching QMD exactly::

    {"timestamp": ..., "fixture": ..., "results": [...], "summary": {...}}

where each result is ``{id, query, type, backends: {<name>: {...}}}`` and each
backend carries ``precision_at_k, recall, recall_at_1, recall_at_3,
recall_at_5, mrr, f1, hits_at_k, matched_files, unmatched_expected_files,
total_expected, latency_ms, top_files``.

QMD's backends are ``bm25 / vector / hybrid / full``; MemBukkit's are its
retrieval modes ``dense / rerank / chain / decompose``. They occupy the same
slot in the schema, so the summary tables line up column for column.

Scoring is :mod:`benchmarks.common.qmd_compat`, a port of QMD's ``score.ts``,
quirks included: ``precision_at_k`` divides by ``min(k, len(expected))`` and
unsuffixed ``recall`` is computed over the whole result list. Reproducing QMD's
numbers means reproducing its definitions, not correcting them.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Dict, List, Optional, Sequence

from benchmarks.common import qmd_compat

SUMMARY_FIELDS = (
    "avg_precision", "avg_recall", "avg_recall_at_1", "avg_recall_at_3",
    "avg_recall_at_5", "avg_mrr", "avg_f1", "avg_latency_ms",
)
_PER_QUERY_TO_SUMMARY = {
    "avg_precision": "precision_at_k",
    "avg_recall": "recall",
    "avg_recall_at_1": "recall_at_1",
    "avg_recall_at_3": "recall_at_3",
    "avg_recall_at_5": "recall_at_5",
    "avg_mrr": "mrr",
    "avg_f1": "f1",
}


def load_fixture(path: pathlib.Path) -> Dict:
    """Load a QMD-format ``queries.json``."""
    data = json.loads(pathlib.Path(path).read_text())
    if "queries" not in data:
        raise ValueError(f"{path} is not a QMD fixture: no 'queries' key")
    return data


def qmd_uri(collection: str, filename: str) -> str:
    """Render a result path the way QMD reports one."""
    return f"qmd://{collection}/{filename}"


def score_backend(
    top_files: Sequence[str],
    expected_files: Sequence[str],
    top_k: int,
    latency_ms: float,
) -> Dict:
    """One backend's per-query block, field-for-field with QMD's."""
    scored = qmd_compat.score_results(list(top_files), list(expected_files), top_k)
    return {
        **scored,
        "total_expected": len(expected_files),
        "latency_ms": latency_ms,
        "top_files": list(top_files),
    }


def build_report(
    fixture_path: str,
    queries: Sequence[Dict],
    backend_runs: Dict[str, Dict[str, Dict]],
    *,
    timestamp: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    """Assemble a QMD-shaped report.

    ``backend_runs`` maps ``backend -> query_id -> {"top_files", "latency_ms"}``.
    A backend missing a query is simply absent from that result, which is how
    QMD behaves when a backend errors.
    """
    ids = [str(q.get("id", "")) for q in queries]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise ValueError(
            f"fixture has duplicate query ids {dupes}: results are keyed by id, "
            f"so duplicates collapse every affected query into one entry "
            f"carrying whichever ranking was written last."
        )

    results: List[Dict] = []
    for q in queries:
        qid = str(q.get("id", ""))
        expected = q.get("expected_files") or []
        top_k = int(q.get("expected_in_top_k") or 10)
        backends: Dict[str, Dict] = {}
        for name, runs in backend_runs.items():
            run = runs.get(qid)
            if run is None:
                continue
            backends[name] = score_backend(
                run.get("top_files", []), expected, top_k,
                float(run.get("latency_ms", 0.0)))
        results.append({"id": qid, "query": q.get("query", ""),
                        "type": q.get("type", ""), "backends": backends})

    report = {
        "timestamp": timestamp or time.strftime("%Y-%m-%dT%H%M", time.gmtime()),
        "fixture": str(fixture_path),
        "results": results,
        "summary": summarize(results),
    }
    if extra:
        report.update(extra)
    return report


def summarize(results: Sequence[Dict]) -> Dict[str, Dict[str, float]]:
    """Per-backend averages, matching QMD's ``summary`` block."""
    out: Dict[str, Dict[str, float]] = {}
    names: List[str] = []
    for r in results:
        for n in r.get("backends", {}):
            if n not in names:
                names.append(n)
    for name in names:
        rows = [r["backends"][name] for r in results if name in r.get("backends", {})]
        if not rows:
            continue
        out[name] = {
            field: sum(float(r.get(src, 0.0)) for r in rows) / len(rows)
            for field, src in _PER_QUERY_TO_SUMMARY.items()
        }
        out[name]["avg_latency_ms"] = (
            sum(float(r.get("latency_ms", 0.0)) for r in rows) / len(rows))
    return out


def summary_table(reports: Dict[str, Dict]) -> str:
    """Render one or more QMD-shaped reports as a single comparison table."""
    head = (f"{'system':<12}{'backend':<12}{'prec':>8}{'recall':>8}{'R@1':>8}"
            f"{'R@3':>8}{'R@5':>8}{'MRR':>8}{'F1':>8}{'latency':>10}")
    lines = [head, "-" * len(head)]
    for system, report in reports.items():
        for backend, s in (report.get("summary") or {}).items():
            lines.append(
                f"{system:<12}{backend:<12}"
                f"{s.get('avg_precision', 0):>8.3f}{s.get('avg_recall', 0):>8.3f}"
                f"{s.get('avg_recall_at_1', 0):>8.3f}{s.get('avg_recall_at_3', 0):>8.3f}"
                f"{s.get('avg_recall_at_5', 0):>8.3f}{s.get('avg_mrr', 0):>8.3f}"
                f"{s.get('avg_f1', 0):>8.3f}{s.get('avg_latency_ms', 0):>8.0f}ms")
    return "\n".join(lines)


def shared_query_ids(reports: Sequence[Dict]) -> List[str]:
    """Query ids present in every report, so a comparison is like-for-like."""
    if not reports:
        return []
    common = {str(r["id"]) for r in reports[0].get("results", [])}
    for rep in reports[1:]:
        common &= {str(r["id"]) for r in rep.get("results", [])}
    order = [str(r["id"]) for r in reports[0].get("results", [])]
    return [q for q in order if q in common]


def restrict(report: Dict, query_ids: Sequence[str]) -> Dict:
    """Re-summarise a report over a subset of queries."""
    keep = set(query_ids)
    results = [r for r in report.get("results", []) if str(r["id"]) in keep]
    return {**report, "results": results, "summary": summarize(results)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare QMD-format bench reports.")
    ap.add_argument("reports", nargs="+", type=pathlib.Path,
                    help="qmd-format JSON files; label with label=path to rename")
    ap.add_argument("--all-queries", action="store_true",
                    help="skip restricting to queries present in every report")
    args = ap.parse_args()

    loaded: Dict[str, Dict] = {}
    for item in args.reports:
        raw = str(item)
        label, _, path = raw.partition("=")
        if not path:
            label, path = pathlib.Path(raw).stem, raw
        loaded[label] = json.loads(pathlib.Path(path).read_text())

    if not args.all_queries and len(loaded) > 1:
        shared = shared_query_ids(list(loaded.values()))
        counts = {k: len(v.get("results", [])) for k, v in loaded.items()}
        if any(c != len(shared) for c in counts.values()):
            print(f"restricting to {len(shared)} queries present in every report "
                  f"(had {counts})\n")
        loaded = {k: restrict(v, shared) for k, v in loaded.items()}

    print(summary_table(loaded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
