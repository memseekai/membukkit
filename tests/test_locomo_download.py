"""Tests for the LoCoMo data loader's auto-download fallback."""

from __future__ import annotations

import json

import pytest

from membukkit.data import locomo

MINIMAL_LOCOMO = [
    {
        "sample_id": "conv0",
        "conversation": {
            "session_1": [
                {"speaker": "Ada", "text": "I adopted a cat named Miso.", "dia_id": "D1:1"},
                {"speaker": "Ben", "text": "Congrats!", "dia_id": "D1:2"},
            ],
            "session_1_date_time": "1:56 pm on 8 May, 2023",
        },
        "qa": [
            {"question": "What is the cat's name?", "answer": "Miso",
             "evidence": ["D1:1"], "category": 4},
        ],
    }
]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """No locomo10.json in cwd, cache redirected into tmp."""
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "cache" / "locomo10.json"
    monkeypatch.setattr(locomo, "_LOCOMO_CACHE", cache)
    return cache


def test_autodownload_uses_cache_path(isolated, monkeypatch):
    calls = {"n": 0}

    def fake_get(url, timeout=60.0):
        calls["n"] += 1
        assert url == locomo.LOCOMO_URL
        return json.dumps(MINIMAL_LOCOMO).encode()

    monkeypatch.setattr(locomo, "_http_get", fake_get)
    ds = locomo.load_locomo("does_not_exist.json")
    assert calls["n"] == 1
    assert isolated.exists()
    assert len(ds.instances) == 1

    # Second load is served from the cache, no HTTP.
    ds2 = locomo.load_locomo("does_not_exist.json")
    assert calls["n"] == 1
    assert len(ds2.instances) == 1


def test_download_failure_gives_clear_error(isolated, monkeypatch):
    def fake_get(url, timeout=60.0):
        raise OSError("network unreachable")

    monkeypatch.setattr(locomo, "_http_get", fake_get)
    with pytest.raises(FileNotFoundError) as exc:
        locomo.load_locomo("does_not_exist.json")
    msg = str(exc.value)
    assert locomo.LOCOMO_URL in msg
    assert "locomo10.json" in msg
    assert not isolated.exists()


def test_non_json_response_is_not_cached(isolated, monkeypatch):
    monkeypatch.setattr(locomo, "_http_get", lambda url, timeout=60.0: b"<html>404</html>")
    with pytest.raises(FileNotFoundError):
        locomo.load_locomo("does_not_exist.json")
    assert not isolated.exists()


def test_explicit_path_skips_download(isolated, tmp_path, monkeypatch):
    data_file = tmp_path / "my_locomo.json"
    data_file.write_text(json.dumps(MINIMAL_LOCOMO))

    def boom(url, timeout=60.0):
        raise AssertionError("HTTP must not be called when an explicit path exists")

    monkeypatch.setattr(locomo, "_http_get", boom)
    ds = locomo.load_locomo(str(data_file))
    assert len(ds.instances) == 1
