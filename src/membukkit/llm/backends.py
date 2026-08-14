"""LLM backend abstraction layer."""

from __future__ import annotations
from typing import Optional, Protocol


class LLMBackend(Protocol):
    def __call__(self, prompt: str) -> str: ...


class OpenAIBackend:
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        timeout: float = 120.0,
        reasoning_effort: str = "",
    ):
        import os

        self.model = model
        self.temperature = temperature
        # Per-request timeout so one hung socket can't stall a whole concurrent
        # eval (the SDK default is ~10 min, which looks like a frozen run).
        self.timeout = timeout
        # Graduated effort for reasoning models (gpt-5.x / o-series); only sent
        # to models that don't take a temperature, so a gpt-4o judge sharing the
        # env var is unaffected. Mirrors COMPAT_REASONING_EFFORT on LocalBackend.
        self.reasoning_effort = (
            reasoning_effort or os.environ.get("OPENAI_REASONING_EFFORT", "")
        ).strip()
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(timeout=self.timeout, max_retries=2)
        return self._client

    def _supports_temperature(self) -> bool:
        m = self.model.lower()
        return not (
            m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")
        )

    def __call__(self, prompt: str) -> str:
        client = self._get_client()
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": self.timeout,
        }
        if self._supports_temperature():
            kwargs["temperature"] = self.temperature
        elif self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        response = client.chat.completions.create(**kwargs)
        _record_openai_usage(response)
        return response.choices[0].message.content


class AnthropicBackend:
    def __init__(self, model: str = "claude-sonnet-4-20250514", temperature: float = 0.3):
        self.model = model
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def __call__(self, prompt: str) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        _record_anthropic_usage(response)
        return response.content[0].text


class GoogleBackend:
    """Google GenAI (Gemini / Gemma) via the ``google-genai`` SDK.

    Auth uses ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) from the environment
    unless an explicit ``api_key`` is passed. Works for hosted Gemma models such
    as ``gemma-4-26b-a4b-it`` as well as ``gemini-*`` models.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        api_key: str = "",
        max_retries: int = 6,
        vertexai: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        # Vertex AI Express mode (genai.Client(vertexai=True, api_key=...)) has
        # separate, higher quotas than the AI Studio Gemini API, but only serves
        # Gemini models (Gemma requires a deployed Model Garden endpoint).
        self.vertexai = vertexai
        # The google-genai SDK does NOT retry 429/5xx by default (it raises
        # immediately), so under concurrent eval load rate-limit errors would
        # otherwise bubble up and get silently turned into wrong answers.
        self.max_retries = max_retries
        self._client = None

    @staticmethod
    def _mask(key: str) -> str:
        if not key:
            return "<empty>"
        if len(key) <= 10:
            return f"{key[:2]}…{key[-2:]} (len={len(key)})"
        return f"{key[:6]}…{key[-4:]} (len={len(key)})"

    def _resolve_key(self):
        """Return (key, source) exactly as the SDK would pick it up, so we can
        report which credential is actually in use."""
        import os

        if self.api_key:
            return self.api_key, "explicit api_key"
        env_vars = (
            ("VERTEX_API_KEY", "GOOGLE_API_KEY")
            if self.vertexai
            else ("GEMINI_API_KEY", "GOOGLE_API_KEY")
        )
        for var in env_vars:
            val = os.getenv(var)
            if val:
                return val, f"env:{var}"
        return "", "none"

    _logged_keys: set = set()

    def _get_client(self):
        if self._client is None:
            import logging

            from google import genai

            key, source = self._resolve_key()
            sig = (self.model, key, source, self.vertexai)
            if sig not in GoogleBackend._logged_keys:
                GoogleBackend._logged_keys.add(sig)
                mode = "vertex" if self.vertexai else "aistudio"
                logging.getLogger(__name__).info(
                    f"GoogleBackend[{mode}] model={self.model} using API key "
                    f"{self._mask(key)} [source={source}]"
                )
            if self.vertexai:
                self._client = (
                    genai.Client(vertexai=True, api_key=key) if key else genai.Client(vertexai=True)
                )
            else:
                self._client = (
                    genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
                )
        return self._client

    @staticmethod
    def _is_retriable(err: Exception) -> bool:
        msg = str(err)
        return any(
            tok in msg
            for tok in (
                "429",
                "RESOURCE_EXHAUSTED",
                "500",
                "503",
                "UNAVAILABLE",
                "INTERNAL",
                "DEADLINE_EXCEEDED",
            )
        )

    def __call__(self, prompt: str) -> str:
        import random
        import time

        client = self._get_client()
        try:
            from google.genai import types

            config = types.GenerateContentConfig(temperature=self.temperature)
        except Exception:  # older/newer SDK: fall back to a plain dict
            config = {"temperature": self.temperature}

        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                _record_google_usage(response)
                return getattr(response, "text", "") or ""
            except Exception as e:  # noqa: BLE001
                if attempt >= self.max_retries or not self._is_retriable(e):
                    raise
                time.sleep(delay + random.uniform(0, 1.0))
                delay = min(delay * 2, 60.0)
        return ""  # unreachable, keeps type-checkers happy


class LocalBackend:
    """OpenAI-compatible chat backend.

    Serves double duty: a local server with no auth (Ollama, vLLM, LM Studio ->
    ``api_key="not-needed"``) or a hosted pay-per-token provider (Fireworks,
    Together, DeepInfra, OpenRouter, Groq, Vertex's OpenAI endpoint) by passing a
    real ``base_url`` + ``api_key``. Hosted providers rate-limit under concurrent
    eval load, so 429/5xx are retried with jittered exponential backoff.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "local-model",
        temperature: float = 0.3,
        api_key: str = "not-needed",
        max_retries: int = 6,
        timeout: Optional[float] = 120.0,
        reasoning_effort: str = "",
        enable_thinking: bool = False,
        extra_body: Optional[dict] = None,
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or "not-needed"
        self.max_retries = max_retries
        # Per-request timeout so a single hung connection can't block the whole
        # concurrent eval. We disable the SDK's own retries (max_retries=0) and
        # do our own jittered backoff below.
        self.timeout = timeout
        # Thinking control for hybrid-reasoning models. Gemma 4 thinking is a
        # boolean (docs: enable_thinking=True/False in the chat template) — there
        # is no graduated budget — so enable_thinking is the canonical switch and
        # is sent as chat_template_kwargs.enable_thinking. reasoning_effort is
        # kept for models that DO honor low/medium/high (e.g. GPT-OSS). In both
        # cases the reasoning trace lands in reasoning_content while the final
        # answer stays in content, so returning content is unaffected.
        self.reasoning_effort = reasoning_effort or ""
        self.enable_thinking = enable_thinking
        self.extra_body = extra_body
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=0,
            )
        return self._client

    @staticmethod
    def _is_retriable(err: Exception) -> bool:
        msg = str(err)
        return any(
            tok in msg
            for tok in (
                "429",
                "RESOURCE_EXHAUSTED",
                "rate limit",
                "Rate limit",
                "500",
                "502",
                "503",
                "504",
                "overloaded",
                "Overloaded",
                "timeout",
                "Timeout",
                "timed out",
                "APITimeoutError",
                "Connection",
            )
        )

    def __call__(self, prompt: str) -> str:
        import random
        import time

        client = self._get_client()
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "timeout": self.timeout,
                }
                if self.reasoning_effort:
                    kwargs["reasoning_effort"] = self.reasoning_effort
                extra_body = dict(self.extra_body) if self.extra_body else {}
                if self.enable_thinking:
                    ctk = dict(extra_body.get("chat_template_kwargs") or {})
                    ctk["enable_thinking"] = True
                    extra_body["chat_template_kwargs"] = ctk
                if extra_body:
                    kwargs["extra_body"] = extra_body
                response = client.chat.completions.create(**kwargs)
                _record_openai_usage(response)
                return response.choices[0].message.content
            except Exception as e:  # noqa: BLE001
                if attempt >= self.max_retries or not self._is_retriable(e):
                    raise
                time.sleep(delay + random.uniform(0, 1.0))
                delay = min(delay * 2, 60.0)
        return ""  # unreachable, keeps type-checkers happy


def _record_openai_usage(response) -> None:
    from membukkit.usage import record_api_usage

    usage = getattr(response, "usage", None)
    if usage is None:
        return
    record_api_usage(
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
    )


def _record_anthropic_usage(response) -> None:
    from membukkit.usage import record_api_usage

    usage = getattr(response, "usage", None)
    if usage is None:
        return
    record_api_usage(
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
    )


def _record_google_usage(response) -> None:
    from membukkit.usage import record_api_usage

    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return
    prompt = getattr(meta, "prompt_token_count", None) or getattr(
        meta, "prompt_tokens", None
    )
    completion = getattr(meta, "candidates_token_count", None) or getattr(
        meta, "output_token_count", None
    )
    record_api_usage(prompt, completion)


def _ollama_base_url() -> str:
    """OpenAI-compatible base URL for a local Ollama server (honors OLLAMA_HOST)."""
    import os

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip()
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/") + "/v1"


def make_llm_backend(backend: str, model: str = "", **kwargs) -> LLMBackend:
    if backend == "openai":
        return OpenAIBackend(model=model or "gpt-4o", **kwargs)
    elif backend == "anthropic":
        return AnthropicBackend(model=model or "claude-sonnet-4-20250514", **kwargs)
    elif backend in ("google", "gemini"):
        return GoogleBackend(model=model or "gemini-2.0-flash", **kwargs)
    elif backend == "vertex":
        # Vertex AI Express mode: reads VERTEX_API_KEY from env (see _resolve_key).
        return GoogleBackend(model=model or "gemini-2.5-flash", vertexai=True, **kwargs)
    elif backend == "ollama":
        # Ollama serves an OpenAI-compatible API locally; reuse LocalBackend.
        return LocalBackend(base_url=_ollama_base_url(), model=model or "llama3", **kwargs)
    elif backend == "local":
        return LocalBackend(model=model or "local-model", **kwargs)
    elif backend == "compat":
        # Any hosted OpenAI-compatible provider (Fireworks, Together, DeepInfra,
        # OpenRouter, Groq, Vertex OpenAI endpoint, ...). base_url + api_key come
        # from env so the model spec stays free of secrets. Thinking control:
        #   COMPAT_ENABLE_THINKING (1/true/yes/on) -> chat_template_kwargs
        #     .enable_thinking (canonical on/off switch; correct for Gemma 4).
        #   COMPAT_REASONING_EFFORT (low/medium/high) -> reasoning_effort, for
        #     models that honor graduated effort (e.g. GPT-OSS).
        import os

        base_url, api_key = _compat_env()
        effort = os.environ.get("COMPAT_REASONING_EFFORT", "").strip()
        enable_thinking = os.environ.get("COMPAT_ENABLE_THINKING", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        # Thinking responses are much longer/slower, so allow the per-request
        # timeout to be tuned via env (default 120s). Set COMPAT_TIMEOUT to
        # "none"/"0"/"off" to disable the timeout entirely (no limit).
        timeout: Optional[float] = 120.0
        _to = os.environ.get("COMPAT_TIMEOUT", "").strip().lower()
        if _to in ("none", "0", "off"):
            timeout = None
        elif _to:
            try:
                timeout = float(_to)
            except ValueError:
                timeout = 120.0
        # Retry budget for transient 429/5xx/timeouts, tunable via env.
        try:
            max_retries = int(os.environ.get("COMPAT_MAX_RETRIES", "").strip() or 6)
        except ValueError:
            max_retries = 6
        return LocalBackend(
            base_url=base_url,
            model=model or "local-model",
            api_key=api_key,
            reasoning_effort=effort,
            enable_thinking=enable_thinking,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Use 'openai', 'anthropic', 'google', "
            f"'ollama', 'local', or 'compat'."
        )


def _compat_env() -> tuple[str, str]:
    """Resolve base_url + api_key for the generic OpenAI-compatible provider.

    Reads ``COMPAT_BASE_URL`` (required) and ``COMPAT_API_KEY`` (falls back to
    ``OPENAI_API_KEY``). Lets ``--reader compat:<model>`` target any hosted
    provider without baking secrets into the CLI args.
    """
    import os

    base_url = os.environ.get("COMPAT_BASE_URL", "").strip()
    if not base_url:
        raise ValueError(
            "compat backend requires COMPAT_BASE_URL to be set (e.g. "
            "https://api.fireworks.ai/inference/v1)."
        )
    api_key = (
        os.environ.get("COMPAT_API_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    return base_url, api_key


_KNOWN_BACKENDS = (
    "openai",
    "anthropic",
    "google",
    "gemini",
    "vertex",
    "ollama",
    "local",
    "compat",
)
_GOOGLE_MODEL_PREFIXES = ("gemma", "gemini")


def resolve_llm(spec: str) -> LLMBackend:
    """Resolve a reader/judge/distiller model string to a backend.

    - ``"backend:model"`` (e.g. ``"google:gemma-4-26b-a4b-it"``,
      ``"ollama:gemma3:27b"``, ``"local:http://host:8000/v1:model"``, or
      ``"compat:accounts/fireworks/models/..."``) -> that backend explicitly.
    - a bare ``gemma-*`` / ``gemini-*`` name -> Google GenAI (so
      ``--reader gemma-4-26b-a4b-it`` just works).
    - any other bare name -> OpenAI (back-compat; e.g. ``gpt-4o-mini``).
    """
    if ":" in spec and spec.split(":", 1)[0] in _KNOWN_BACKENDS:
        return parse_llm_spec(spec)
    if spec.lower().startswith(_GOOGLE_MODEL_PREFIXES):
        return make_llm_backend("google", model=spec)
    return make_llm_backend("openai", model=spec)


def parse_llm_spec(spec: str) -> LLMBackend:
    """Parse a spec string into an LLM backend.

    Formats:
        "openai:gpt-4o-mini"
        "anthropic:claude-sonnet-4-20250514"
        "google:gemma-4-26b-a4b-it"   (alias: "gemini:...")
        "ollama:gemma3:27b"           (local Ollama, OLLAMA_HOST-aware)
        "local:http://localhost:8000/v1:model-name"
        "compat:<model>"              (hosted OpenAI-compatible provider;
                                       COMPAT_BASE_URL + COMPAT_API_KEY from env)
    """
    parts = spec.split(":", maxsplit=1)
    if len(parts) < 2:
        raise ValueError(
            f"Invalid spec '{spec}'. Expected format: 'backend:model' (e.g. 'openai:gpt-4o-mini')"
        )
    backend_name, rest = parts[0], parts[1]

    if backend_name in ("ollama", "compat"):
        # rest is the full model id, which itself may contain ':' (e.g. Ollama
        # tags like gemma3:27b, or provider paths). base_url comes from env.
        return make_llm_backend(backend_name, model=rest)

    if backend_name == "local":
        url_and_model = rest.rsplit(":", maxsplit=1)
        if len(url_and_model) == 2:
            base_url, model = url_and_model
        else:
            base_url, model = rest, "local-model"
        return LocalBackend(base_url=base_url, model=model)

    return make_llm_backend(backend_name, model=rest)
