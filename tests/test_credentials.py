"""Local credentials file + GUI keys API."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMBUKKIT_HOME", str(tmp_path))
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OLLAMA_HOST",
    ):
        monkeypatch.delenv(key, raising=False)

    import membukkit.credentials as cred

    monkeypatch.setattr(cred, "_BOOTSTRAPPED", False)
    monkeypatch.setattr(cred, "_PREEXISTING_ENV", set())
    yield


def test_file_roundtrip_and_mask(tmp_path, monkeypatch):
    from membukkit.credentials import (
        bootstrap_credentials,
        credentials_path,
        key_status,
        set_keys,
    )

    applied, path = set_keys(openai_api_key="sk-test-secret-abcd1234", persist=True)
    assert "OPENAI_API_KEY" in applied
    assert path == str(credentials_path())
    mode = credentials_path().stat().st_mode & 0o777
    assert mode == 0o600 or mode == 0o666  # Windows may ignore chmod

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import membukkit.credentials as cred

    monkeypatch.setattr(cred, "_BOOTSTRAPPED", False)
    loaded = bootstrap_credentials()
    assert "OPENAI_API_KEY" in loaded
    assert os.environ["OPENAI_API_KEY"].endswith("1234")

    status = key_status("openai:gpt-4o-mini")
    assert status["ready"] is True
    assert status["needs"] == "openai"
    assert status["providers"]["openai"]["mask"] == "…1234"
    assert "sk-test" not in str(status)


def test_env_wins_over_file(tmp_path, monkeypatch):
    from membukkit.credentials import bootstrap_credentials, set_keys, write_credentials_file

    write_credentials_file({"OPENAI_API_KEY": "from-file"})
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    import membukkit.credentials as cred

    monkeypatch.setattr(cred, "_BOOTSTRAPPED", False)
    assert bootstrap_credentials() == []
    assert os.environ["OPENAI_API_KEY"] == "from-shell"
    # set_keys still works
    set_keys(openai_api_key="from-gui", persist=False)
    assert os.environ["OPENAI_API_KEY"] == "from-gui"


def test_keys_api(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMBUKKIT_HOME", str(tmp_path))
    from membukkit.service.local_app import create_local_app

    client = TestClient(create_local_app(llm="openai:gpt-4o-mini"))
    r = client.get("/api/settings/keys")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is False
    assert body["needs"] == "openai"
    assert "sk-" not in str(body)

    r = client.put(
        "/api/settings/keys",
        json={"openai_api_key": "sk-live-xxxx9999", "persist": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["providers"]["openai"]["mask"] == "…9999"
    assert body["persisted_to"]
    assert os.environ["OPENAI_API_KEY"].endswith("9999")

    # Full secret never echoed
    assert "sk-live-xxxx9999" not in r.text
