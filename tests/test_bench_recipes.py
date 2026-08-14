"""Tests for the frozen benchmark repro recipes and `membukkit bench --repro`."""

from __future__ import annotations

import argparse
import json
import sys

import pytest

from membukkit.bench.recipes import RECIPES, check_recipe_output, get_recipe
from membukkit.cli import bench, eval_legacy

EXPECTED_IDS = {
    "longmemeval-gpt54",
    "longmemeval-gemma",
    "longmemeval-gpt4o-mini",
    "locomo-mem0",
    "beam-100k-gemma",
    "beam-1m-gemma",
    "beam-10m-gemma",
}

GEMMA = "compat:google/gemma-4-26b-a4b-it"


def _bench_args(argv):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    bench.register(sub)
    return parser.parse_args(["bench", *argv])


# ---------------------------------------------------------------- registry


def test_registry_has_all_recipes():
    assert set(RECIPES) == EXPECTED_IDS


def test_get_recipe_unknown_id_lists_known():
    with pytest.raises(KeyError, match="longmemeval-gpt54"):
        get_recipe("nope")


def test_expected_numbers_frozen():
    assert RECIPES["longmemeval-gpt54"].expected == 0.926
    assert RECIPES["longmemeval-gemma"].expected == 0.888
    assert RECIPES["longmemeval-gpt4o-mini"].expected == 0.820
    assert RECIPES["locomo-mem0"].expected == 0.875
    assert RECIPES["beam-100k-gemma"].expected == 0.535
    assert RECIPES["beam-1m-gemma"].expected == 0.498
    assert RECIPES["beam-10m-gemma"].expected == 0.447


# ------------------------------------------------------- distiller pinning


@pytest.mark.parametrize("recipe", RECIPES.values(), ids=list(RECIPES))
def test_every_argv_pins_the_distiller(recipe):
    """No recipe may fall through to the eval default distill model."""
    assert "--distill-model" in recipe.argv
    idx = recipe.argv.index("--distill-model")
    assert recipe.argv[idx + 1] == recipe.distiller


def test_gpt54_distills_with_gpt54():
    r = RECIPES["longmemeval-gpt54"]
    assert r.distiller == "gpt-5.4"
    idx = r.argv.index("--distill-model")
    assert r.argv[idx + 1] == "gpt-5.4"


@pytest.mark.parametrize(
    "rid", ["longmemeval-gemma", "beam-100k-gemma", "beam-1m-gemma", "beam-10m-gemma"]
)
def test_gemma_recipes_distill_with_gemma(rid):
    r = RECIPES[rid]
    assert r.distiller == GEMMA
    idx = r.argv.index("--distill-model")
    assert r.argv[idx + 1] == GEMMA


@pytest.mark.parametrize("recipe", RECIPES.values(), ids=list(RECIPES))
def test_every_argv_pins_a_per_recipe_distill_cache(recipe):
    assert recipe.distill_cache == f"runs/distill_cache_{recipe.id}.json"
    idx = recipe.argv.index("--distill-cache")
    assert recipe.argv[idx + 1] == recipe.distill_cache


# ------------------------------------- argv validity against the real parser


@pytest.mark.parametrize("recipe", RECIPES.values(), ids=list(RECIPES))
def test_recipe_argv_parses_against_real_eval_parser(recipe, monkeypatch):
    """Every frozen flag must exist in the real `membukkit eval` argparse.

    Builds the actual parser (via eval_legacy.main with the eval command
    stubbed out) and parses the recipe argv; an unknown/invalid flag exits
    with SystemExit(2) and fails the test.
    """
    captured = {}
    monkeypatch.setattr(eval_legacy, "_eval_cmd", lambda args: captured.update(args=args))
    monkeypatch.setattr(sys, "argv", ["membukkit", "eval", *recipe.argv])
    eval_legacy.main()

    args = captured["args"]
    assert args.dataset == recipe.dataset
    assert args.distill_model == recipe.distiller
    assert args.distill_cache == recipe.distill_cache
    if recipe.dataset == "beam":
        assert args.beam_scale in ("100K", "1M", "10M")
        assert args.deep_broad is True
    if recipe.id == "locomo-mem0":
        assert args.reader_protocol == "mem0"
        assert args.judge_protocol == "mem0"
        assert args.locomo_drop_categories == "5"


# ------------------------------------------------------------ --check logic


def test_check_acc_pass_and_fail(tmp_path):
    """Derived from the recipe's own tolerance so retuning the band cannot
    silently invert this test's meaning."""
    recipe = RECIPES["longmemeval-gpt54"]
    summary = tmp_path / "e2e_summary.json"
    inside = round(recipe.expected - recipe.tolerance / 2, 4)
    outside = round(recipe.expected - recipe.tolerance * 2, 4)

    summary.write_text(json.dumps({"overall": {"coremem_union": {"acc": inside}}}))
    passed, measured, path = check_recipe_output(recipe, str(tmp_path))
    assert passed and measured == inside and path == summary

    summary.write_text(json.dumps({"overall": {"coremem_union": {"acc": outside}}}))
    passed, measured, _ = check_recipe_output(recipe, str(tmp_path))
    assert not passed and measured == outside


def test_check_rejects_incomplete_run(tmp_path):
    """A --lite subset lands in the same output dir; grading it against a
    full-run number would report a meaningless verdict."""
    recipe = RECIPES["longmemeval-gpt4o-mini"]
    summary = tmp_path / "e2e_summary.json"

    summary.write_text(
        json.dumps({"overall": {"coremem_union": {"acc": 1.0, "n": 3}}})
    )
    with pytest.raises(ValueError, match="covers 3 question"):
        check_recipe_output(recipe, str(tmp_path))

    # a complete run with the same shape grades normally
    summary.write_text(
        json.dumps(
            {"overall": {"coremem_union": {"acc": recipe.expected, "n": recipe.expected_n}}}
        )
    )
    passed, _, _ = check_recipe_output(recipe, str(tmp_path))
    assert passed


def test_check_without_n_still_grades(tmp_path):
    """Summaries that report no count say nothing about completeness."""
    recipe = RECIPES["longmemeval-gpt4o-mini"]
    (tmp_path / "e2e_summary.json").write_text(
        json.dumps({"overall": {"coremem_union": {"acc": recipe.expected}}})
    )
    passed, _, _ = check_recipe_output(recipe, str(tmp_path))
    assert passed


def test_check_beam_avg_pass_and_fail(tmp_path):
    recipe = RECIPES["beam-100k-gemma"]  # expected 0.535, tol 0.02
    summary = tmp_path / "beam_summary.json"

    summary.write_text(json.dumps({"scores": {"average": 0.545}}))
    passed, measured, path = check_recipe_output(recipe, str(tmp_path))
    assert passed and measured == 0.545 and path == summary

    summary.write_text(json.dumps({"scores": {"average": 0.60}}))
    passed, _, _ = check_recipe_output(recipe, str(tmp_path))
    assert not passed


def test_check_missing_summary_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="membukkit bench --repro"):
        check_recipe_output(RECIPES["locomo-mem0"], str(tmp_path / "nowhere"))


def test_cli_check_pass_and_fail_exit_codes(tmp_path, capsys):
    out = tmp_path / "run"
    out.mkdir()
    (out / "e2e_summary.json").write_text(
        json.dumps({"overall": {"coremem_union": {"acc": 0.875}}})
    )

    args = _bench_args(["--repro", "locomo-mem0", "--check", "--output-dir", str(out)])
    bench.cmd_bench(args)
    assert "[PASS]" in capsys.readouterr().out

    (out / "e2e_summary.json").write_text(
        json.dumps({"overall": {"coremem_union": {"acc": 0.50}}})
    )
    with pytest.raises(SystemExit) as exc:
        bench.cmd_bench(args)
    assert exc.value.code == 1
    assert "[FAIL]" in capsys.readouterr().out


# ------------------------------------------------------------ --repro env


def test_repro_missing_env_fails_clearly(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = _bench_args(["--repro", "longmemeval-gpt54", "--yes"])
    with pytest.raises(SystemExit) as exc:
        bench.cmd_bench(args)
    assert "OPENAI_API_KEY" in str(exc.value)
    assert "longmemeval-gpt54" in str(exc.value)


def test_repro_gemma_requires_compat_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("COMPAT_API_KEY", raising=False)
    args = _bench_args(["--repro", "longmemeval-gemma", "--yes"])
    with pytest.raises(SystemExit) as exc:
        bench.cmd_bench(args)
    msg = str(exc.value)
    assert "COMPAT_BASE_URL" in msg and "COMPAT_API_KEY" in msg


def test_repro_unknown_id_fails_clearly():
    args = _bench_args(["--repro", "not-a-recipe"])
    with pytest.raises(SystemExit, match="unknown recipe"):
        bench.cmd_bench(args)


def test_repro_builds_argv_and_sets_beam_env_default(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("COMPAT_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("COMPAT_API_KEY", "ck-test")
    monkeypatch.delenv("MEMBUKKIT_DISTILL_MAX_TURN_CHARS", raising=False)

    calls = {}
    monkeypatch.setattr(bench, "_run_eval", lambda argv: calls.update(argv=argv))

    args = _bench_args(["--repro", "beam-100k-gemma", "--lite", "--yes"])
    bench.cmd_bench(args)

    argv = calls["argv"]
    assert argv[0] == "eval"
    # recipe output dir injected; lite subset appended
    oi = argv.index("--output-dir")
    assert argv[oi + 1] == "results/bench/beam-100k-gemma"
    assert argv[argv.index("--max-instances") + 1] == "3"
    # distiller pinned in the invoked command line
    assert argv[argv.index("--distill-model") + 1] == GEMMA
    # BEAM turn-cap env default applied because it was unset
    import os

    assert os.environ["MEMBUKKIT_DISTILL_MAX_TURN_CHARS"] == "4000"
    # the full equivalent eval command is printed
    assert "harness: membukkit eval" in capsys.readouterr().out


def test_repro_respects_output_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls = {}
    monkeypatch.setattr(bench, "_run_eval", lambda argv: calls.update(argv=argv))

    args = _bench_args(
        ["--repro", "longmemeval-gpt4o-mini", "--lite", "--yes",
         "--output-dir", str(tmp_path / "custom")]
    )
    bench.cmd_bench(args)
    argv = calls["argv"]
    assert argv[argv.index("--output-dir") + 1] == str(tmp_path / "custom")


# ------------------------------------------------------------------ --list


def test_list_shows_all_recipes(capsys):
    args = _bench_args(["--list"])
    bench.cmd_bench(args)
    out = capsys.readouterr().out
    for rid in EXPECTED_IDS:
        assert rid in out
    assert "OPENAI_API_KEY" in out
    assert "expected" in out
