"""`membukkit bench` — one-command benchmark presets and frozen repro recipes.

Thin wrappers over the full research harness (``membukkit eval``) so anyone
can reproduce the README numbers without learning its ~70 flags. ``--lite``
runs a small subset first; every run prints a cost estimate before spending.

Two modes:

- free-form presets: ``membukkit bench longmemeval --lite`` (pick your own
  reader/judge/encoder);
- frozen repro recipes: ``membukkit bench --list`` /
  ``membukkit bench --repro <id>`` — the exact pinned configs
  (reader/distiller/judge/encoder/routing flags) behind each claimed number,
  verified against the original run artifacts. ``--repro <id> --check``
  compares a completed run's summary against the frozen number.
"""

from __future__ import annotations

import os
import sys

from membukkit.usage import PRICE_PER_M_INPUT as _PRICE_PER_M

# Rough distillation+reading+judging input volumes per full run (tokens).
# Sources: measured runs with gpt-4o-mini distiller + reader on each dataset.
_EST_TOKENS = {
    "longmemeval": 60_000_000,
    "locomo": 9_000_000,
    "beam-100K": 2_500_000,
    "beam-500K": 20_000_000,
    "beam-1M": 40_000_000,
    "beam-10M": 120_000_000,
}

_DEFAULT_OUTPUT_DIR = "results/bench"


def _price_for(model_spec: str) -> float | None:
    spec = model_spec.lower()
    for key, price in _PRICE_PER_M.items():
        if key in spec:
            return price
    return None


def _estimate(dataset_key: str, model_spec: str, lite_fraction: float) -> str:
    tokens = _EST_TOKENS.get(dataset_key)
    if tokens is None:
        return "cost estimate unavailable for this preset"
    tokens = int(tokens * lite_fraction)
    price = _price_for(model_spec)
    if price is None:
        return f"~{tokens / 1e6:.0f}M LLM input tokens (unknown $ rate for {model_spec!r})"
    return f"~{tokens / 1e6:.0f}M LLM input tokens, roughly ${tokens / 1e6 * price:.2f} with {model_spec}"


def _run_eval(argv: list[str]) -> None:
    from membukkit.cli import eval_legacy

    sys.argv = ["membukkit"] + argv
    eval_legacy.main()


def _cmd_list(args) -> None:
    from membukkit.bench.recipes import RECIPES

    print("frozen repro recipes (membukkit bench --repro <id>):\n")
    header = f"{'id':<24} {'dataset':<12} {'expected':>8}  {'metric':<8} {'reader':<34}"
    print(header)
    print("-" * len(header))
    for r in RECIPES.values():
        print(f"{r.id:<24} {r.dataset:<12} {r.expected:>8.3f}  {r.metric:<8} {r.reader:<34}")
        print(f"{'':<24} distiller: {r.distiller}")
        print(f"{'':<24} judge:     {r.judge}")
        print(f"{'':<24} encoder:   {r.encoder}")
        env_bits = []
        for var in r.required_env:
            env_bits.append(f"{var}[{'set' if os.environ.get(var) else 'MISSING'}]")
        for var, val in r.env_defaults.items():
            env_bits.append(f"{var}={os.environ.get(var, val)}")
        print(f"{'':<24} env:       {' '.join(env_bits)}")
        print(f"{'':<24} cost:      {_estimate(r.estimate_key, r.distiller, 1.0)}")
        print()


def _cmd_repro(args) -> None:
    from membukkit.bench.recipes import check_recipe_output, get_recipe

    try:
        recipe = get_recipe(args.repro)
    except KeyError as e:
        raise SystemExit(str(e).strip('"'))

    output_dir = args.output_dir or recipe.output_subdir

    def _check(exit_on_fail: bool) -> None:
        try:
            passed, measured, summary_path = check_recipe_output(recipe, output_dir)
        except FileNotFoundError as e:
            raise SystemExit(f"check failed: {e}")
        verdict = "PASS" if passed else "FAIL"
        print(
            f"[{verdict}] {recipe.id}: measured {recipe.metric}={measured:.3f} vs "
            f"expected {recipe.expected:.3f} (tolerance ±{recipe.tolerance:.2f}) "
            f"from {summary_path}"
        )
        print(
            "note: LLM readers/judges are nondeterministic — this is a sanity "
            "band, not an exact-match test."
        )
        if not passed and exit_on_fail:
            raise SystemExit(1)

    if args.check:
        # Standalone verification pass over an existing output dir (no run).
        _check(exit_on_fail=True)
        return

    # Recipe env defaults (e.g. the BEAM distiller turn cap) apply only when
    # the user hasn't set the variable themselves.
    for var, val in recipe.env_defaults.items():
        if not os.environ.get(var):
            os.environ[var] = val
            print(f"env: {var}={val} (recipe default)")
    missing = [v for v in recipe.required_env if not os.environ.get(v)]
    if missing:
        raise SystemExit(
            f"recipe {recipe.id!r} requires environment variables that are not set: "
            f"{', '.join(missing)}. Export them (or add them to .env) and retry."
        )

    argv = ["eval", *recipe.argv, "--output-dir", output_dir]
    lite_fraction = 1.0
    if args.lite:
        argv += ["--max-instances", str(args.lite_n)]
        lite_fraction = 0.06

    print(f"repro recipe: {recipe.id} — {recipe.title}"
          + (" (lite subset)" if args.lite else " (FULL RUN)"))
    print(f"expected: {recipe.metric}={recipe.expected:.3f} (±{recipe.tolerance:.2f})")
    print(f"cost: {_estimate(recipe.estimate_key, recipe.distiller, lite_fraction)}")
    print(f"harness: membukkit {' '.join(argv)}\n")
    if not args.yes and not args.lite:
        reply = input("full run — proceed? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("aborted (use --lite for a cheap subset, or --yes to skip this prompt)")
            return

    _run_eval(argv)

    # Lite subsets are not comparable to the frozen full-run numbers.
    if not args.lite:
        _check(exit_on_fail=False)


def _cmd_preset(args) -> None:
    dataset_key = args.dataset if args.dataset != "beam" else f"beam-{args.scale}"
    lite_fraction = 0.06 if args.lite else 1.0
    output_dir = args.output_dir or _DEFAULT_OUTPUT_DIR

    argv = ["eval", "--dataset", args.dataset, "--methods", "coremem_union",
            "--reader", args.reader, "--distill-model", args.reader,
            "--output-dir", output_dir,
            "--distill-cache", f"distill_cache_{dataset_key.replace('-', '_')}.json"]
    if args.dataset == "beam":
        argv += ["--beam-scale", args.scale, "--official-judge",
                 "--agg-top-k", "50", "--agg-scan-budget", "1.0",
                 "--deep-top-k", "60", "--deep-scan-budget", "1.0", "--deep-broad"]
    if args.dataset == "locomo":
        argv += ["--locomo-path", args.locomo_path]
    if args.lite:
        argv += ["--max-instances", str(args.lite_n)]
    if args.judge:
        argv += ["--judge", args.judge]
    if args.encoder:
        argv += ["--encoder", args.encoder]
    if args.extra:
        argv += args.extra

    print(f"benchmark: {dataset_key}" + (" (lite subset)" if args.lite else " (FULL RUN)"))
    print(f"cost: {_estimate(dataset_key, args.reader, lite_fraction)}")
    print(f"harness: membukkit {' '.join(argv)}\n")
    if not args.yes and not args.lite:
        reply = input("full run — proceed? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("aborted (use --lite for a cheap subset, or --yes to skip this prompt)")
            return

    _run_eval(argv)


def cmd_bench(args) -> None:
    if args.list_recipes:
        _cmd_list(args)
        return
    if args.repro:
        _cmd_repro(args)
        return
    if not args.dataset:
        raise SystemExit(
            "membukkit bench: pick a preset dataset (longmemeval / locomo / beam), "
            "or use --list / --repro <id> for the frozen repro recipes."
        )
    _cmd_preset(args)


def register(sub) -> None:
    p = sub.add_parser(
        "bench",
        help="Run a benchmark preset or a frozen repro recipe",
        description="One-command benchmark presets over the full eval harness. "
        "Use --lite for a fast, cheap subset; drop it to reproduce full numbers. "
        "Use --list / --repro <id> for the frozen, artifact-verified repro recipes "
        "behind every claimed number.",
    )
    p.add_argument("dataset", nargs="?", choices=["longmemeval", "locomo", "beam"])
    p.add_argument("--list", dest="list_recipes", action="store_true",
                   help="list the frozen repro recipes (config, env, cost)")
    p.add_argument("--repro", metavar="ID", default=None,
                   help="run a frozen repro recipe (see --list)")
    p.add_argument("--check", action="store_true",
                   help="with --repro: verify an existing output dir against the "
                   "recipe's expected number (PASS/FAIL, nonzero exit on FAIL)")
    p.add_argument("--lite", action="store_true", help="small subset (fast, cheap)")
    p.add_argument("--lite-n", type=int, default=3,
                   help="conversations/haystacks in the lite subset (default 3)")
    p.add_argument("--scale", default="100K", choices=["100K", "500K", "1M", "10M"],
                   help="BEAM scale (beam preset only)")
    p.add_argument("--reader", default="gpt-4o-mini", help="reader+distiller LLM (presets)")
    p.add_argument("--judge", default=None, help="override judge LLM (presets)")
    p.add_argument("--encoder", default=None, help="override encoder spec (presets)")
    p.add_argument("--locomo-path", default="locomo10.json")
    p.add_argument("--output-dir", default=None,
                   help=f"output directory (default: {_DEFAULT_OUTPUT_DIR} for presets, "
                   "results/bench/<id> for --repro)")
    p.add_argument("--yes", action="store_true", help="skip the full-run confirmation")
    p.add_argument("extra", nargs="*", help="extra flags passed through to `membukkit eval`")
    p.set_defaults(func=cmd_bench)
