# Benchmarks: reproducing the numbers

Every score MemBukkit claims is backed by a **frozen recipe**, a registry entry that pins the
reader, distiller, judge, and encoder to the exact models the number was measured with. One
command reruns it:

```bash
membukkit bench --repro <recipe-id>
```

## What we claim

| Recipe id | Benchmark | Reader / Distiller | Judge | Encoder | Expected |
|---|---|---|---|---|---|
| `longmemeval-gpt54` | LongMemEval-S | gpt-5.4 | gpt-4o (official) | `openai:text-embedding-3-large@1536` | **92.6%** |
| `longmemeval-gemma` | LongMemEval-S | gemma-4-26b | gpt-4o (official) | `openai:text-embedding-3-large@1536` | **88.8%** |
| `longmemeval-gpt4o-mini` | LongMemEval-S | gpt-4o-mini | gpt-4o (official) | `biencoder_v1` (fine-tuned) | **82.0%** |
| `locomo-mem0` | LoCoMo (Mem0 protocol) | gpt-4o-mini | gpt-4o-mini (Mem0 judge) | `biencoder_v1` | **87.5%** |
| `beam-100k-gemma` | BEAM 100K | gemma-4-26b | official (gpt-4.1-mini) | `openai:text-embedding-3-large@1536` | **0.535** |
| `beam-1m-gemma` | BEAM 1M | gemma-4-26b | official (gpt-4.1-mini) | `openai:text-embedding-3-large@1536` | **0.498** |
| `beam-10m-gemma` | BEAM 10M | gemma-4-26b | official (gpt-4.1-mini) | `openai:text-embedding-3-large@1536` | **0.447** |

In every recipe the distiller is the same model as the reader, each recipe pins its own
distillation cache, because distillation quality materially affects the score (see
[the distiller note](#the-distiller-is-part-of-the-recipe) below).

> **Reproduction is a band, not bit-identity**
>
> `--check` compares your run against the expected score within a tolerance band:
> **±0.03** for accuracy metrics (LongMemEval, LoCoMo) and **±0.02** for BEAM averages.
> A run inside the band is a successful reproduction.
>
> The band absorbs three things a rerun cannot control: reader nondeterminism, judge
> nondeterminism, and drift in the hosted models a recipe pins by name (the model behind
> `gpt-4o-mini` keeps moving even though the string does not). For scale, the binomial
> standard error on a 500-question benchmark is already ~1.8 points. The band is set wide
> enough to pass a healthy rerun and still tight enough to catch what it is actually for:
> a broken setup, wrong encoder, missing extra, stale distill cache, wrong judge, which
> costs tens of points, not three.
>
> `--check` grades complete runs only. A `--lite` subset written to the same output
> directory is rejected rather than scored against a full-run number.

## Who judges what {#who-judges-what}

LongMemEval ships an official judge: gpt-4o, with the benchmark's own prompts,
validated at ~97% agreement with human graders. Every MemBukkit number above is
scored by it. That is the whole reason the numbers are comparable to anything.

Several published results are higher than 92.6, and they are graded by their
own authors rather than by the official judge:

| System | Reported | Who graded it |
|---|---|---|
| OMEGA | 95.4 | GPT-4.1, used as **both** the answering and the grading model |
| Mem0 Cloud | 94.4 | own GPT-5 judge |
| **MemBukkit** | **92.6** | **official gpt-4o judge** |
| Hindsight | 91.4 | official prompts, judge model swapped to GPT-OSS-120B |
| Mem0 OSS | 91.0 | own GPT-5 judge |
| Supermemory | 85.2 | official judge |
| Zep | 71.2 | official judge |
| Full-context reading | 60.2 | official judge |

Restricted to systems the official judge scored, MemBukkit is the highest
published result on LongMemEval-S. We are not claiming those other numbers are
wrong; a self-graded score simply is not measuring the same thing, and a model
grading its own answers is not measuring much at all. If you want to compare
without taking anyone's word for it, `membukkit bench --repro` runs the official
judge on your machine.

## Quickstart

List the recipes, each entry shows the pinned models, the expected score, the environment
variables it needs, and a cost estimate:

```bash
membukkit bench --list
```

The recommended first run is the cheapest recipe on a lite subset. It costs cents and finishes in
minutes:

```bash
export OPENAI_API_KEY=sk-...
membukkit bench --repro longmemeval-gpt4o-mini --lite
```

`--lite` runs a few haystacks of the same frozen recipe, same models, same settings, as a smoke
test that your keys, dataset download, and encoder weights all work. Lite scores are **not**
comparable to the expected numbers (the subset is too small); drop `--lite` for the real thing.

A full run prints its cost estimate and asks for confirmation before spending anything
(`--yes` skips the prompt). When it finishes, verify the score:

```bash
membukkit bench --repro longmemeval-gpt4o-mini --check
```

`--check` reads the finished run's summary from `results/bench/longmemeval-gpt4o-mini/` and
reports whether the score falls inside the tolerance band.

## Prerequisites

- **`OPENAI_API_KEY`**: required by every recipe (embeddings and/or judges are OpenAI models).
- **`COMPAT_BASE_URL` + `COMPAT_API_KEY`**: required by the gemma recipes
  (`longmemeval-gemma`, `beam-*-gemma`). Any OpenAI-compatible host serving
  `google/gemma-4-26b-a4b-it` works; the published numbers were measured against DeepInfra.
- **Datasets auto-download** on first use: LongMemEval from the HF Hub, BEAM from the BEAM
  GitHub repository's raw files, and LoCoMo from the official repo.
- **Model weights auto-download**: recipes using the fine-tuned `biencoder_v1` encoder fetch it
  from the MemseekAI HF org on first use (cached under `~/.membukkit/models`).
- BEAM recipes set `MEMBUKKIT_DISTILL_MAX_TURN_CHARS=4000` automatically, you do not need to
  export it.

## The recipes

Cost figures below are ballpark LLM **input-token volumes** for a full run, from measured runs
(the same estimates the CLI prints before asking for confirmation). Dollar figures depend on your
provider's rates; the gpt-4o-mini figures use $0.15 per 1M input tokens.

### `longmemeval-gpt4o-mini`, LongMemEval-S, 82.0%

The recommended starting point: cheapest full recipe, no compat host needed.

```bash
membukkit bench --repro longmemeval-gpt4o-mini          # full run
membukkit bench --repro longmemeval-gpt4o-mini --check  # verify afterwards
```

- **Needs**: `OPENAI_API_KEY`.
- **Cost/time**: ~60M input tokens ≈ $9 at gpt-4o-mini rates, plus the gpt-4o judge over 500
  questions. Expect hours, not minutes (distillation of ~500 long haystacks dominates).
- **Success**: `acc` in `e2e_summary.json` within 82.0% ± 3 points.

### `longmemeval-gpt54`, LongMemEval-S, 92.6%

The headline number.

```bash
membukkit bench --repro longmemeval-gpt54
```

- **Needs**: `OPENAI_API_KEY`.
- **Cost/time**: same ~60M-token volume, but at gpt-5.4 rates the dollar cost is substantially
  higher than the gpt-4o-mini recipe. Hours.
- **Success**: `acc` within 92.6% ± 3 points.

### `longmemeval-gemma`, LongMemEval-S, 88.8%

Open-weights reader/distiller via an OpenAI-compatible host.

```bash
export COMPAT_BASE_URL=https://api.deepinfra.com/v1/openai
export COMPAT_API_KEY=...
membukkit bench --repro longmemeval-gemma
```

- **Needs**: `OPENAI_API_KEY` (judge + embeddings) and `COMPAT_BASE_URL`/`COMPAT_API_KEY`
  (gemma).
- **Cost/time**: ~60M input tokens at your compat host's gemma rate; typically cheaper than
  gpt-4o-mini on DeepInfra. Hours.
- **Success**: `acc` within 88.8% ± 3 points.

### `locomo-mem0`, LoCoMo under the Mem0 protocol, 87.5%

Follows the Mem0 evaluation protocol, including its gpt-4o-mini judge, so the number is
comparable with Mem0's published table. The dataset downloads from the official LoCoMo repo on
first use.

```bash
membukkit bench --repro locomo-mem0
```

- **Needs**: `OPENAI_API_KEY`.
- **Cost/time**: ~9M input tokens ≈ $1.35 at gpt-4o-mini rates. Under an hour, usually.
- **Success**: `acc` within 87.5% ± 3 points.

### `beam-100k-gemma` / `beam-1m-gemma` / `beam-10m-gemma`, BEAM, official scorer

BEAM haystacks range from 100K to 10M tokens of conversation; all three recipes use the official
BEAM judge (gpt-4.1-mini) and report the official average over the 9 ability categories.

```bash
membukkit bench --repro beam-100k-gemma    # 0.535 expected
membukkit bench --repro beam-1m-gemma     # 0.498 expected
membukkit bench --repro beam-10m-gemma    # 0.447 expected
```

- **Needs**: `OPENAI_API_KEY` and `COMPAT_BASE_URL`/`COMPAT_API_KEY`.
- **Cost/time**: ~2.5M input tokens at 100K, ~40M at 1M, ~120M at 10M. The 10M scale ingests
  roughly ten million tokens of conversation per haystack, read the printed estimate carefully
  before confirming. 100K finishes quickly; 10M is a long-running job.
- **Success**: `average` in `beam_summary.json` within ±0.02 of the expected value.

## Where results land

Each recipe writes to its own directory:

```
results/bench/<recipe-id>/
├── e2e_summary.json      # LongMemEval / LoCoMo — headline field: "acc"
├── beam_summary.json     # BEAM — headline field: "average"
└── ...                   # per-question verdicts, ability breakdowns, retrieval traces
```

`--check` reads the summary from this directory, so it works any time after a run finishes, you
do not need to keep the original terminal session.

## The distiller is part of the recipe

MemBukkit stores each session twice: verbatim turns and LLM-distilled atomic facts. The
distillation model determines what the atomic lane contains, which materially affects the final
score, a recipe run with a different distiller is a different experiment. Each recipe therefore
pins its own distill model **and its own distillation cache file**, so recipes never contaminate
each other, and rerunning a recipe reuses its cache (re-runs after the first are dominated by
reading and judging, not distillation).

## Frozen recipes vs free-form presets

The free-form presets still exist for quick experiments:

```bash
membukkit bench longmemeval --lite --reader gpt-4o-mini
```

These let you swap readers, judges, and encoders freely, useful for trying your own models, but
they make no fidelity claim. **Only `--repro <recipe-id>` pins everything the published number
depends on.** If you want to compare against the table above, use `--repro`. The full research
surface (routing policies, scan budgets, ablations) remains available under `membukkit eval --help`.

## Troubleshooting

**"Missing environment variable" at startup.** The recipe told you exactly which one. Gemma
recipes need `COMPAT_BASE_URL` and `COMPAT_API_KEY` in addition to `OPENAI_API_KEY`; run
`membukkit bench --list` to see each recipe's requirements before starting.

**LoCoMo download fails.** The dataset is fetched from the official LoCoMo repository on first
use. If your network blocks GitHub raw content, download `locomo10.json` manually and place it
where the error message indicates.

**A run was interrupted.** Rerun the same `--repro` command. Distillation results are cached in
the recipe's distill cache and completed tasks are cached at the task level, so the rerun skips
everything already done and resumes where it stopped.

**My score is outside the band.** First check you ran the frozen recipe (`--repro`), not a
free-form preset, a different distiller, judge, or encoder shifts scores by more than the
tolerance. If the recipe is right, rerun `--check` against a *complete* run (an interrupted run's
summary covers only a subset of questions). Persistent gaps beyond the band on a complete frozen
run are worth an issue report.

**Encoder weights fail to download.** `biencoder_v1` recipes fetch weights from the MemseekAI HF
org. See [Install, model weights](install.md#model-weights) for the resolution order and
offline options.
