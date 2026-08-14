"""Frozen reproduction recipes for every claimed benchmark number.

Each :class:`Recipe` pins the EXACT reader/distiller/judge/encoder and routing
flags that produced an artifact-backed claim, verified field-by-field against
the original run summary JSONs (the ``config`` block written by
``membukkit eval``). The distiller is pinned explicitly in every argv via
``--distill-model`` — a recipe must never fall through to the eval harness's
default distill model.

Deviations found in the artifacts vs the working notes, frozen here as the
artifacts dictate:

- ``longmemeval-gemma`` ran with ``--reader-prompts v1`` and aggregation
  routing ONLY (no ``--deep-top-k``/``--deep-scan-budget``).
- All BEAM runs had ``official_judge: false``: BEAM is scored by the
  benchmark's own vendored judge (gpt-4.1-mini @ temp 0, official prompts),
  which the harness selects automatically for ``--dataset beam``, so the
  LongMemEval ``--official-judge`` flag is not part of those recipes.
- BEAM runs used ``--reader-prompts v1`` (the default), not v2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_GEMMA = "compat:google/gemma-4-26b-a4b-it"
_OPENAI_EMBED = "openai:text-embedding-3-large@1536"

# Verification tolerances: a sanity band around the frozen number, not an
# exact-match test. A repro has to absorb three sources of movement — reader
# nondeterminism, judge nondeterminism, and provider-side drift in the hosted
# models a recipe pins by name (the model behind "gpt-4o-mini" is not frozen
# just because the string is).
#
# ±0.03 on accuracy is chosen from both ends:
#   floor   — a full 500-question LongMemEval rerun in Aug 2026 measured 0.798
#             against the frozen 0.820, a 2.2-point gap and ~1.2x the binomial
#             standard error at n=500 (~1.8 points). A tighter band fails on
#             healthy runs, which is worse than useless: it teaches people to
#             ignore the check.
#   ceiling — the band must still catch what this check exists for, which is a
#             broken setup (wrong encoder, missing extra, stale distill cache,
#             wrong judge). Those cost 10-40 points, not 3.
_TOLERANCE = {"acc": 0.03, "beam_avg": 0.02}

# Required (value None) or set-if-missing (value str) environment variables.
_ENV_OPENAI: Dict[str, Optional[str]] = {"OPENAI_API_KEY": None}
_ENV_GEMMA: Dict[str, Optional[str]] = {
    "OPENAI_API_KEY": None,  # judge + remote embeddings
    "COMPAT_BASE_URL": None,  # OpenAI-compatible host serving the gemma model
    "COMPAT_API_KEY": None,
}
# BEAM turns average ~1,900 chars vs the 600-char LongMemEval-tuned default
# cap; the original runs distilled with a 4,000-char cap.
_ENV_BEAM_GEMMA: Dict[str, Optional[str]] = {
    **_ENV_GEMMA,
    "MEMBUKKIT_DISTILL_MAX_TURN_CHARS": "4000",
}


@dataclass(frozen=True)
class Recipe:
    """A frozen, named repro of one artifact-backed benchmark claim."""

    id: str
    title: str
    dataset: str  # longmemeval | locomo | beam
    expected: float
    metric: str  # "acc" | "beam_avg"
    description: str
    reader: str
    distiller: str
    judge: str
    encoder: str
    argv: List[str] = field(default_factory=list)  # `membukkit eval` flags
    env: Dict[str, Optional[str]] = field(default_factory=dict)
    distill_cache: str = ""
    output_subdir: str = ""
    # Questions a complete run must score, so `--check` can refuse to grade a
    # partial run (notably `--lite`, which writes its subset summary to the same
    # output dir). 0 disables the guard for datasets whose size is not pinned.
    expected_n: int = 0

    @property
    def required_env(self) -> List[str]:
        return [k for k, v in self.env.items() if v is None]

    @property
    def env_defaults(self) -> Dict[str, str]:
        return {k: v for k, v in self.env.items() if v is not None}

    @property
    def tolerance(self) -> float:
        return _TOLERANCE[self.metric]

    @property
    def estimate_key(self) -> str:
        """Dataset key for the `membukkit bench` cost-estimate tables."""
        if self.dataset == "beam" and "--beam-scale" in self.argv:
            scale = self.argv[self.argv.index("--beam-scale") + 1]
            return f"beam-{scale}"
        return self.dataset


def _recipe(
    id: str,
    *,
    title: str,
    dataset: str,
    expected: float,
    metric: str,
    description: str,
    reader: str,
    distiller: str,
    judge: str,
    encoder: str,
    argv: List[str],
    env: Dict[str, Optional[str]],
    expected_n: int = 0,
) -> Recipe:
    distill_cache = f"runs/distill_cache_{id}.json"
    return Recipe(
        expected_n=expected_n,
        id=id,
        title=title,
        dataset=dataset,
        expected=expected,
        metric=metric,
        description=description,
        reader=reader,
        distiller=distiller,
        judge=judge,
        encoder=encoder,
        argv=[*argv, "--distill-cache", distill_cache],
        env=env,
        distill_cache=distill_cache,
        output_subdir=f"results/bench/{id}",
    )


def _beam_recipe(scale: str, expected: float) -> Recipe:
    # Frozen from results/beam/gemma_{scale}_routed_hints/e2e_summary.json.
    # The e2e config records judge=gpt-4o with official_judge=false; for
    # --dataset beam the harness replaces that parser default with BEAM's own
    # vendored judge (gpt-4.1-mini @ temp 0, official prompts + aggregation).
    rid = f"beam-{scale.lower()}-gemma"
    return _recipe(
        rid,
        title=f"BEAM {scale} — Gemma routed+hints",
        dataset="beam",
        expected=expected,
        metric="beam_avg",
        description=(
            f"BEAM {scale} split, coremem_union with aggregation + broad deep "
            "routing and per-category answer hints; Gemma 4 26B reads and "
            "distills, official vendored BEAM judge (gpt-4.1-mini) scores. "
            "Requires MEMBUKKIT_DISTILL_MAX_TURN_CHARS=4000 (set automatically "
            "if unset)."
        ),
        reader=_GEMMA,
        distiller=_GEMMA,
        judge="gpt-4.1-mini (official BEAM judge)",
        encoder=_OPENAI_EMBED,
        argv=[
            "--dataset", "beam",
            "--beam-scale", scale,
            "--methods", "coremem_union",
            "--bucket-mode", "topic",
            "--rerank-select", "hybrid",
            "--encoder", _OPENAI_EMBED,
            "--reader", _GEMMA,
            "--judge", "gpt-4o",
            "--reader-prompts", "v1",
            "--agg-top-k", "50",
            "--agg-scan-budget", "1.0",
            "--deep-top-k", "60",
            "--deep-scan-budget", "1.0",
            "--deep-broad",
            "--distill-model", _GEMMA,
        ],
        env=_ENV_BEAM_GEMMA,
    )


_ALL_RECIPES: Tuple[Recipe, ...] = (
    # Frozen from results/longmemeval/deeproute_54targeted_full/e2e_summary.json.
    _recipe(
        "longmemeval-gpt54",
        title="LongMemEval — GPT-5.4 deep routing",
        dataset="longmemeval",
        expected=0.926,
        metric="acc",
        description=(
            "Headline LongMemEval number: coremem_union with aggregation + deep "
            "routing, v2 reader prompts, GPT-5.4 reads AND distills, official "
            "gpt-4o judge, remote OpenAI embeddings."
        ),
        reader="gpt-5.4",
        distiller="gpt-5.4",
        judge="gpt-4o",
        encoder=_OPENAI_EMBED,
        argv=[
            "--dataset", "longmemeval",
            "--methods", "coremem_union",
            "--bucket-mode", "topic",
            "--rerank-select", "hybrid",
            "--encoder", _OPENAI_EMBED,
            "--reader", "gpt-5.4",
            "--judge", "gpt-4o",
            "--official-judge",
            "--reader-prompts", "v2",
            "--agg-top-k", "50",
            "--agg-scan-budget", "1.0",
            "--deep-top-k", "60",
            "--deep-scan-budget", "1.0",
            "--distill-model", "gpt-5.4",
        ],
        env=_ENV_OPENAI,
        expected_n=500,
    ),
    # Frozen from results/longmemeval/gemma_agg_routing/e2e_summary.json.
    # NOTE: this run used v1 reader prompts and aggregation routing only
    # (no deep-routing flags), unlike the gpt-5.4 recipe.
    _recipe(
        "longmemeval-gemma",
        title="LongMemEval — Gemma agg routing",
        dataset="longmemeval",
        expected=0.888,
        metric="acc",
        description=(
            "Open-weights LongMemEval number: coremem_union with aggregation "
            "routing (no deep routing), v1 reader prompts, Gemma 4 26B reads "
            "AND distills via an OpenAI-compatible host, official gpt-4o judge, "
            "remote OpenAI embeddings."
        ),
        reader=_GEMMA,
        distiller=_GEMMA,
        judge="gpt-4o",
        encoder=_OPENAI_EMBED,
        argv=[
            "--dataset", "longmemeval",
            "--methods", "coremem_union",
            "--bucket-mode", "topic",
            "--rerank-select", "hybrid",
            "--encoder", _OPENAI_EMBED,
            "--reader", _GEMMA,
            "--judge", "gpt-4o",
            "--official-judge",
            "--reader-prompts", "v1",
            "--agg-top-k", "50",
            "--agg-scan-budget", "1.0",
            "--distill-model", _GEMMA,
        ],
        env=_ENV_GEMMA,
        expected_n=500,
    ),
    # Frozen from results/longmemeval/v2/e2e_summary_hybrid.json.
    _recipe(
        "longmemeval-gpt4o-mini",
        title="LongMemEval — gpt-4o-mini + fine-tuned encoder",
        dataset="longmemeval",
        expected=0.820,
        metric="acc",
        description=(
            "Budget LongMemEval number: coremem_union with the fine-tuned "
            "biencoder_v1 encoder, standard 0.3 scan budget (no agg/deep "
            "routing levers), gpt-4o-mini reads AND distills, official gpt-4o "
            "judge."
        ),
        reader="gpt-4o-mini",
        distiller="gpt-4o-mini",
        judge="gpt-4o",
        encoder="biencoder_v1",
        argv=[
            "--dataset", "longmemeval",
            "--methods", "coremem_union",
            "--bucket-mode", "topic",
            "--rerank-select", "hybrid",
            "--scan-budget", "0.3",
            "--encoder", "biencoder_v1",
            "--reader", "gpt-4o-mini",
            "--judge", "gpt-4o",
            "--official-judge",
            "--reader-prompts", "v1",
            "--distill-model", "gpt-4o-mini",
        ],
        env=_ENV_OPENAI,
        expected_n=500,
    ),
    # Frozen from results/locomo/runC_mem0_reader_mem0_judge.json.
    _recipe(
        "locomo-mem0",
        title="LoCoMo — Mem0 protocol",
        dataset="locomo",
        expected=0.875,
        metric="acc",
        description=(
            "LoCoMo cross-dataset number under the Mem0 reader+judge protocol "
            "(gpt-4o-mini for both, NOT the LongMemEval official judge), "
            "fine-tuned biencoder_v1 encoder, adversarial category 5 dropped."
        ),
        reader="gpt-4o-mini",
        distiller="gpt-4o-mini",
        judge="gpt-4o-mini",
        encoder="biencoder_v1",
        argv=[
            "--dataset", "locomo",
            "--locomo-drop-categories", "5",
            "--methods", "coremem_union",
            "--bucket-mode", "topic",
            "--rerank-select", "hybrid",
            "--encoder", "biencoder_v1",
            "--reader-protocol", "mem0",
            "--judge-protocol", "mem0",
            "--reader", "gpt-4o-mini",
            "--judge", "gpt-4o-mini",
            "--distill-model", "gpt-4o-mini",
        ],
        env=_ENV_OPENAI,
    ),
    _beam_recipe("100K", 0.535),
    _beam_recipe("1M", 0.498),
    _beam_recipe("10M", 0.447),
)

RECIPES: Dict[str, Recipe] = {r.id: r for r in _ALL_RECIPES}


def get_recipe(recipe_id: str) -> Recipe:
    try:
        return RECIPES[recipe_id]
    except KeyError:
        known = ", ".join(sorted(RECIPES))
        raise KeyError(f"unknown recipe {recipe_id!r}; known recipes: {known}") from None


def check_recipe_output(recipe: Recipe, output_dir: str) -> Tuple[bool, float, Path]:
    """Compare a completed run's summary against the recipe's frozen number.

    Returns ``(passed, measured, summary_path)``; raises FileNotFoundError with
    a clear message when the summary artifact is missing (run not completed, or
    wrong --output-dir).
    """
    out = Path(output_dir)
    if recipe.metric == "beam_avg":
        summary_path = out / "beam_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"no {summary_path} — run `membukkit bench --repro {recipe.id}` "
                "to completion first (or point --output-dir at the run's output)."
            )
        measured = float(json.loads(summary_path.read_text())["scores"]["average"])
    else:
        summary_path = out / "e2e_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"no {summary_path} — run `membukkit bench --repro {recipe.id}` "
                "to completion first (or point --output-dir at the run's output)."
            )
        overall = json.loads(summary_path.read_text())["overall"]
        entry = overall.get("coremem_union") or next(iter(overall.values()))
        measured = float(entry["acc"])
        # A `--lite` run writes its subset summary to the same output dir, so
        # grading whatever is on disk would score 3 questions against a 500-
        # question claim. Refuse instead of reporting a meaningless verdict.
        # Only enforce when the summary actually reports a count; a summary
        # without one says nothing about completeness, so grade it as before.
        n = entry.get("n")
        if recipe.expected_n and n is not None and int(n) < recipe.expected_n:
            raise ValueError(
                f"{summary_path} covers {n} question(s), but {recipe.id} scores "
                f"{recipe.expected_n}. This looks like a --lite or interrupted run; "
                f"--check only grades a complete one. Rerun without --lite "
                f"(`membukkit bench --repro {recipe.id}`), then check again."
            )
    passed = abs(measured - recipe.expected) <= recipe.tolerance
    return passed, measured, summary_path
