"""QMD's scoring semantics, ported from ``src/bench/score.ts``.

Kept separate from :mod:`benchmarks.common.metrics` on purpose: QMD's scorer
does two things that differ from the textbook definitions, and reproducing its
numbers means reproducing the quirks rather than quietly "fixing" them.

1. ``precision_at_k`` divides by ``min(k, len(expected))``, not by ``k``. With
   one expected file and k=10 a single hit therefore scores 1.0, where standard
   precision@10 would score 0.1.
2. ``recall`` (unsuffixed) is computed over the *entire* result list, not the
   top k, so it answers "was it found at all" rather than "was it found early".

QMD also reports recall at 1/3/5 only, and has no nDCG. Anything beyond that in
our reports is labelled as an extension.

Upstream: https://github.com/tobi/qmd (see benchmarks/qmd/fixture/MANIFEST.json
for the pinned commit).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence


def normalize_path(p: str) -> str:
    """Port of QMD ``normalizePath``: strip ``qmd://<collection>/``, lowercase, trim slashes."""
    if p.startswith("qmd://"):
        without_scheme = p[len("qmd://") :]
        slash = without_scheme.find("/")
        p = without_scheme[slash + 1 :] if slash >= 0 else without_scheme
    return p.lower().strip("/")


def paths_match(result: str, expected: str) -> bool:
    """Port of QMD ``pathsMatch``: equality, or either path being a suffix of the other."""
    nr, ne = normalize_path(result), normalize_path(expected)
    if nr == ne:
        return True
    return nr.endswith(ne) or ne.endswith(nr)


def _hits_within(results: Sequence[str], expected: Sequence[str], k: int) -> int:
    top = results[:k]
    return sum(1 for e in expected if any(paths_match(r, e) for r in top))


def qmd_precision_at_k(results: Sequence[str], expected: Sequence[str], k: int) -> float:
    """QMD's precision: hits@k / min(k, len(expected)). Not textbook precision@k."""
    denom = min(k, len(expected))
    return _hits_within(results, expected, k) / denom if denom > 0 else 0.0


def score_results(
    results: Sequence[str],
    expected: Sequence[str],
    top_k: int,
) -> Dict[str, object]:
    """Port of QMD ``scoreResults``. Field names match the TypeScript output."""
    hits_at_k = _hits_within(results, expected, top_k)

    matched: List[str] = []
    unmatched: List[str] = []
    for e in expected:
        if any(paths_match(r, e) for r in results):
            matched.append(e)
        else:
            unmatched.append(e)

    mrr = 0.0
    for i, r in enumerate(results):
        if any(paths_match(r, e) for e in expected):
            mrr = 1.0 / (i + 1)
            break

    n_expected = len(expected)
    precision = qmd_precision_at_k(results, expected, top_k)
    recall = len(matched) / n_expected if n_expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision_at_k": precision,
        "recall": recall,
        "recall_at_1": (_hits_within(results, expected, 1) / n_expected) if n_expected else 0.0,
        "recall_at_3": (_hits_within(results, expected, 3) / n_expected) if n_expected else 0.0,
        "recall_at_5": (_hits_within(results, expected, 5) / n_expected) if n_expected else 0.0,
        "mrr": mrr,
        "f1": f1,
        "hits_at_k": hits_at_k,
        "matched_files": matched,
        "unmatched_expected_files": unmatched,
    }


def match_any(results: Iterable[str], expected: Iterable[str]) -> List[str]:
    """Result paths that match at least one expected path, for reporting."""
    exp = list(expected)
    return [r for r in results if any(paths_match(r, e) for e in exp)]
