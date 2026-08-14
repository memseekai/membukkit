"""LLM backend resolution — routing only (clients are lazy, no network/SDK)."""

from __future__ import annotations

from membukkit.llm.backends import (
    AnthropicBackend,
    GoogleBackend,
    LocalBackend,
    OpenAIBackend,
    make_llm_backend,
    parse_llm_spec,
    resolve_llm,
)


def test_bare_gemma_name_routes_to_google():
    be = resolve_llm("gemma-4-26b-a4b-it")
    assert isinstance(be, GoogleBackend)
    assert be.model == "gemma-4-26b-a4b-it"


def test_bare_gemini_name_routes_to_google():
    assert isinstance(resolve_llm("gemini-2.0-flash"), GoogleBackend)


def test_explicit_google_spec():
    be = resolve_llm("google:gemma-4-26b-a4b-it")
    assert isinstance(be, GoogleBackend)
    assert be.model == "gemma-4-26b-a4b-it"


def test_gemini_alias_spec():
    assert isinstance(resolve_llm("gemini:gemma-4-26b-a4b-it"), GoogleBackend)


def test_bare_openai_name_stays_openai():
    be = resolve_llm("gpt-4o-mini")
    assert isinstance(be, OpenAIBackend)
    assert be.model == "gpt-4o-mini"


def test_explicit_anthropic_spec():
    be = resolve_llm("anthropic:claude-sonnet-4-20250514")
    assert isinstance(be, AnthropicBackend)
    assert be.model == "claude-sonnet-4-20250514"


def test_local_spec_with_url():
    be = resolve_llm("local:http://localhost:8000/v1:my-model")
    assert isinstance(be, LocalBackend)
    assert be.base_url == "http://localhost:8000/v1"
    assert be.model == "my-model"


def test_make_llm_backend_google_default_and_alias():
    assert isinstance(make_llm_backend("google"), GoogleBackend)
    be = make_llm_backend("gemini", model="gemma-4-26b-a4b-it")
    assert isinstance(be, GoogleBackend)
    assert be.model == "gemma-4-26b-a4b-it"


def test_parse_llm_spec_google():
    be = parse_llm_spec("google:gemma-4-26b-a4b-it")
    assert isinstance(be, GoogleBackend)
    assert be.model == "gemma-4-26b-a4b-it"


def test_ollama_spec_keeps_full_tag_and_local_url():
    # The tag itself contains a colon (gemma3:27b) and must survive intact.
    be = resolve_llm("ollama:gemma3:27b")
    assert isinstance(be, LocalBackend)
    assert be.model == "gemma3:27b"
    assert be.base_url == "http://localhost:11434/v1"


def test_ollama_honors_ollama_host(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://192.168.1.5:11434")
    be = resolve_llm("ollama:llama3")
    assert isinstance(be, LocalBackend)
    assert be.base_url == "http://192.168.1.5:11434/v1"
    assert be.model == "llama3"


def test_make_llm_backend_ollama_default():
    assert isinstance(make_llm_backend("ollama"), LocalBackend)


def test_compat_spec_reads_env(monkeypatch):
    monkeypatch.setenv("COMPAT_BASE_URL", "https://api.fireworks.ai/inference/v1")
    monkeypatch.setenv("COMPAT_API_KEY", "sk-test-123")
    be = resolve_llm("compat:accounts/fireworks/models/gemma-4-26b")
    assert isinstance(be, LocalBackend)
    assert be.base_url == "https://api.fireworks.ai/inference/v1"
    assert be.api_key == "sk-test-123"
    assert be.model == "accounts/fireworks/models/gemma-4-26b"


def test_compat_api_key_falls_back_to_openai_key(monkeypatch):
    monkeypatch.setenv("COMPAT_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.delenv("COMPAT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")
    be = resolve_llm("compat:google/gemma-2-27b-it")
    assert isinstance(be, LocalBackend)
    assert be.api_key == "sk-openai-fallback"


def test_compat_requires_base_url(monkeypatch):
    import pytest

    monkeypatch.delenv("COMPAT_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="COMPAT_BASE_URL"):
        resolve_llm("compat:some-model")


def test_compat_reasoning_effort_from_env(monkeypatch):
    monkeypatch.setenv("COMPAT_BASE_URL", "https://api.deepinfra.com/v1/openai")
    monkeypatch.setenv("COMPAT_API_KEY", "sk-x")
    monkeypatch.setenv("COMPAT_REASONING_EFFORT", "medium")
    be = resolve_llm("compat:google/gemma-4-26B-A4B-it")
    assert isinstance(be, LocalBackend)
    assert be.reasoning_effort == "medium"


def test_compat_reasoning_effort_default_empty(monkeypatch):
    monkeypatch.setenv("COMPAT_BASE_URL", "https://api.deepinfra.com/v1/openai")
    monkeypatch.setenv("COMPAT_API_KEY", "sk-x")
    monkeypatch.delenv("COMPAT_REASONING_EFFORT", raising=False)
    be = resolve_llm("compat:google/gemma-4-26B-A4B-it")
    assert isinstance(be, LocalBackend)
    assert be.reasoning_effort == ""


def test_compat_enable_thinking_from_env(monkeypatch):
    monkeypatch.setenv("COMPAT_BASE_URL", "https://api.deepinfra.com/v1/openai")
    monkeypatch.setenv("COMPAT_API_KEY", "sk-x")
    monkeypatch.setenv("COMPAT_ENABLE_THINKING", "1")
    be = resolve_llm("compat:google/gemma-4-26B-A4B-it")
    assert isinstance(be, LocalBackend)
    assert be.enable_thinking is True


def test_compat_enable_thinking_default_off(monkeypatch):
    monkeypatch.setenv("COMPAT_BASE_URL", "https://api.deepinfra.com/v1/openai")
    monkeypatch.setenv("COMPAT_API_KEY", "sk-x")
    monkeypatch.delenv("COMPAT_ENABLE_THINKING", raising=False)
    be = resolve_llm("compat:google/gemma-4-26B-A4B-it")
    assert isinstance(be, LocalBackend)
    assert be.enable_thinking is False
