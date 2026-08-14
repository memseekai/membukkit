"""Bench GUI API tests: recipe listing + subprocess run lifecycle.

Runs never invoke the real harness: the subprocess command is monkeypatched
to quick python scripts that write fixture summaries (or sleep/fail), and the
working directory is redirected to tmp_path so nothing lands in the repo.
"""

from __future__ import annotations

import sys
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import membukkit.service.local_app as local_app  # noqa: E402
from membukkit.bench.recipes import RECIPES  # noqa: E402

EXPECTED_IDS = {
    "longmemeval-gpt54",
    "longmemeval-gemma",
    "longmemeval-gpt4o-mini",
    "locomo-mem0",
    "beam-100k-gemma",
    "beam-1m-gemma",
    "beam-10m-gemma",
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMBUKKIT_HOME", str(tmp_path / "home"))
    # Fresh in-memory registry per test; bench subprocesses run in tmp_path.
    monkeypatch.setattr(local_app, "_BENCH_RUNS", {})
    monkeypatch.setattr(local_app, "_bench_root", lambda: tmp_path)
    yield TestClient(local_app.create_local_app())
    for run in local_app._BENCH_RUNS.values():
        if run["proc"].poll() is None:
            run["proc"].kill()
            run["proc"].wait()


def _fake_command(monkeypatch, script: str):
    monkeypatch.setattr(
        local_app, "_bench_command", lambda recipe_id, lite: [sys.executable, "-c", script]
    )


def _wait_done(client, run_id: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/bench/runs/{run_id}").json()
        if body["status"] != "running":
            return body
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} still running after {timeout}s")


# ---------------------------------------------------------------- recipes


def test_recipes_lists_all_with_env_flags(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("COMPAT_API_KEY", raising=False)

    recipes = client.get("/api/bench/recipes").json()["recipes"]
    assert {r["id"] for r in recipes} == EXPECTED_IDS

    by_id = {r["id"]: r for r in recipes}
    gpt54 = by_id["longmemeval-gpt54"]
    assert gpt54["expected"] == 0.926 and gpt54["metric"] == "acc"
    assert gpt54["distiller"] == "gpt-5.4"
    assert gpt54["cli_command"] == "membukkit bench --repro longmemeval-gpt54 --yes"
    assert gpt54["env"] == [{"name": "OPENAI_API_KEY", "set": True}]
    assert "tokens" in gpt54["cost_estimate"]

    # env flags reflect os.environ at request time, per variable
    gemma_env = {e["name"]: e["set"] for e in by_id["longmemeval-gemma"]["env"]}
    assert gemma_env == {
        "OPENAI_API_KEY": True,
        "COMPAT_BASE_URL": False,
        "COMPAT_API_KEY": False,
    }


# ------------------------------------------------------------- validation


def test_start_unknown_recipe_404(client):
    r = client.post("/api/bench/runs", json={"recipe_id": "not-a-recipe", "lite": True})
    assert r.status_code == 404


def test_start_missing_env_400_names_vars(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("COMPAT_API_KEY", raising=False)

    r = client.post("/api/bench/runs", json={"recipe_id": "longmemeval-gemma", "lite": True})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "COMPAT_BASE_URL" in detail and "COMPAT_API_KEY" in detail
    assert "OPENAI_API_KEY" not in detail  # only the missing ones are named
    assert not local_app._BENCH_RUNS  # nothing was started


def test_unknown_run_id_404(client):
    assert client.get("/api/bench/runs/deadbeef").status_code == 404


# ---------------------------------------------------------------- lifecycle


def test_full_run_lifecycle_pass(client, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # Quick fake bench: write a passing e2e summary into the recipe output
    # dir (relative to cwd, like the real CLI) and exit 0.
    _fake_command(
        monkeypatch,
        "import json, pathlib;"
        "out = pathlib.Path('results/bench/longmemeval-gpt54');"
        "out.mkdir(parents=True, exist_ok=True);"
        "(out / 'e2e_summary.json').write_text("
        "json.dumps({'overall': {'coremem_union': {'acc': 0.921}}}));"
        "print('fake bench run complete')",
    )

    r = client.post("/api/bench/runs", json={"recipe_id": "longmemeval-gpt54", "lite": False})
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    body = _wait_done(client, run_id)
    assert body["status"] == "done"
    assert body["recipe_id"] == "longmemeval-gpt54" and body["lite"] is False
    assert any("fake bench run complete" in line for line in body["log_tail"])
    # summary parsed, verdict computed within the recipe tolerance
    assert body["result"] == {
        "measured": 0.921,
        "expected": 0.926,
        "tolerance": RECIPES["longmemeval-gpt54"].tolerance,
        "passed": True,
        "smoke": False,
    }
    # the log persisted where the CLI convention puts it
    assert (tmp_path / "results/bench/longmemeval-gpt54/run.log").exists()

    runs = client.get("/api/bench/runs").json()["runs"]
    assert [(x["run_id"], x["status"]) for x in runs] == [(run_id, "done")]


def test_full_run_out_of_band_fails(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _fake_command(
        monkeypatch,
        "import json, pathlib;"
        "out = pathlib.Path('results/bench/locomo-mem0');"
        "out.mkdir(parents=True, exist_ok=True);"
        "(out / 'e2e_summary.json').write_text("
        "json.dumps({'overall': {'coremem_union': {'acc': 0.5}}}))",
    )
    run_id = client.post(
        "/api/bench/runs", json={"recipe_id": "locomo-mem0", "lite": False}
    ).json()["run_id"]
    body = _wait_done(client, run_id)
    assert body["status"] == "done"
    assert body["result"]["measured"] == 0.5 and body["result"]["passed"] is False


def test_lite_run_is_smoke_without_verdict(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _fake_command(
        monkeypatch,
        "import json, pathlib;"
        "out = pathlib.Path('results/bench/longmemeval-gpt4o-mini');"
        "out.mkdir(parents=True, exist_ok=True);"
        "(out / 'e2e_summary.json').write_text("
        "json.dumps({'overall': {'coremem_union': {'acc': 0.67}}}))",
    )
    run_id = client.post(
        "/api/bench/runs", json={"recipe_id": "longmemeval-gpt4o-mini", "lite": True}
    ).json()["run_id"]
    body = _wait_done(client, run_id)
    assert body["status"] == "done" and body["lite"] is True
    # lite subsets are smoke runs: score reported, never PASS/FAIL
    assert body["result"]["smoke"] is True
    assert body["result"]["passed"] is None
    assert body["result"]["measured"] == 0.67


def test_failed_subprocess_reports_failed(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _fake_command(monkeypatch, "import sys; print('boom'); sys.exit(1)")
    run_id = client.post(
        "/api/bench/runs", json={"recipe_id": "longmemeval-gpt54", "lite": True}
    ).json()["run_id"]
    body = _wait_done(client, run_id)
    assert body["status"] == "failed"
    assert "result" not in body
    assert any("boom" in line for line in body["log_tail"])


def test_duplicate_live_run_409(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _fake_command(monkeypatch, "import time; time.sleep(30)")

    r1 = client.post("/api/bench/runs", json={"recipe_id": "longmemeval-gpt54", "lite": True})
    assert r1.status_code == 200
    assert client.get(f"/api/bench/runs/{r1.json()['run_id']}").json()["status"] == "running"

    # same recipe while live -> refused
    r2 = client.post("/api/bench/runs", json={"recipe_id": "longmemeval-gpt54", "lite": False})
    assert r2.status_code == 409

    # a different recipe is fine
    r3 = client.post("/api/bench/runs", json={"recipe_id": "locomo-mem0", "lite": True})
    assert r3.status_code == 200

    # once the first run is gone, the recipe can be started again
    run = local_app._BENCH_RUNS[r1.json()["run_id"]]
    run["proc"].kill()
    run["proc"].wait()
    r4 = client.post("/api/bench/runs", json={"recipe_id": "longmemeval-gpt54", "lite": True})
    assert r4.status_code == 200
