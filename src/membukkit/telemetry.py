"""Dependency-optional observability facade over Pydantic Logfire / OpenTelemetry.

Every module calls `from membukkit import telemetry` and uses `telemetry.span(...)`,
`telemetry.timed(...)`, `telemetry.histogram(...)` etc. without caring whether
`logfire` is installed or configured:

  - `logfire` not installed  -> the import is guarded; all helpers are no-ops.
  - installed but not configured -> Logfire's own API no-ops safely.
  - configured -> spans/metrics flow to Logfire cloud (LOGFIRE_TOKEN) and/or any
    OTLP collector (OTEL_EXPORTER_OTLP_ENDPOINT). Vendor-neutral.

PII: by default we never put raw fact/query text or LLM message content into
telemetry. `configure(capture_content=True)` flips that on for debug only;
`capture_content()` gates the few places that would attach user text.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

_logfire: Any = None
try:  # pragma: no cover - import guard
    import logfire as _logfire_mod

    _logfire = _logfire_mod
except Exception:  # logfire not installed
    pass

logger = logging.getLogger(__name__)

_CONFIGURED = False
_CAPTURE_CONTENT = False
_WARNED: set = set()


def _warn_once(key: str, msg: str) -> None:
    """Log a telemetry failure at WARNING, once per key (avoids per-request spam).

    Telemetry that was explicitly enabled must never fail *silently* — but a
    call that fails every request should not flood the log either.
    """
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(msg, exc_info=True)


# Attribute keys that may carry user text — added to Logfire's scrubbing patterns.
_SCRUB_PATTERNS = ["memory_text", "fact_text", "question", "answer_text", "content"]


def is_enabled() -> bool:
    """True when logfire is importable AND configure() has run."""
    return _logfire is not None and _CONFIGURED


def capture_content() -> bool:
    """Whether raw user text / LLM content may be attached to telemetry."""
    return _CAPTURE_CONTENT


def configure(
    service_name: str = "membukkit",
    environment: Optional[str] = None,
    capture_content: bool = False,
    console: bool = False,
    **kwargs: Any,
) -> bool:
    """Configure Logfire once.

    Returns True when configured, and False *only* when logfire isn't installed
    (the dependency-optional no-op). If logfire IS installed, a configuration
    failure propagates rather than silently disabling telemetry — telemetry that
    was explicitly enabled must not fail silently.

    Export auto-detects: Logfire cloud when LOGFIRE_TOKEN is set, an OTLP collector
    when OTEL_EXPORTER_OTLP_ENDPOINT is set, both if both. If NEITHER is set (and
    console is off), telemetry is configured but nothing is exported — we log a
    WARNING so that "enabled but no data" is never a silent surprise.
    """
    global _CONFIGURED, _CAPTURE_CONTENT
    _CAPTURE_CONTENT = bool(capture_content)
    if _logfire is None:
        logger.warning(
            "telemetry requested but 'logfire' is not installed — install "
            "membukkit[observability] (or [service]) to enable it; telemetry is OFF"
        )
        return False
    if _CONFIGURED:
        return True

    scrubbing = None
    if hasattr(_logfire, "ScrubbingOptions"):
        scrubbing = _logfire.ScrubbingOptions(extra_patterns=_SCRUB_PATTERNS)
    # Do NOT swallow: an explicit configure() that fails should raise loudly.
    _logfire.configure(
        service_name=service_name,
        environment=environment,
        send_to_logfire=kwargs.pop("send_to_logfire", "if-token-present"),
        console=(kwargs.pop("console", None) if console else False),
        scrubbing=scrubbing,
        **kwargs,
    )
    _CONFIGURED = True

    if not (
        os.environ.get("LOGFIRE_TOKEN") or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or console
    ):
        logger.warning(
            "telemetry configured but NO export destination is set — spans/metrics "
            "are created and then dropped. Set LOGFIRE_TOKEN (Logfire cloud) or "
            "OTEL_EXPORTER_OTLP_ENDPOINT (OTLP collector) to see data."
        )

    try:
        _logfire.instrument_system_metrics()
    except Exception:
        _warn_once("system_metrics", "logfire.instrument_system_metrics() failed")
    return _CONFIGURED


# --------------------------------------------------------------------- spans
class _NoopSpan:
    """Stand-in span when telemetry is disabled; matches the bits we use."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def set_attribute(self, *_a, **_k):
        pass

    def set_attributes(self, *_a, **_k):
        pass

    def record_exception(self, *_a, **_k):
        pass

    def set_level(self, *_a, **_k):
        pass


_NOOP_SPAN = _NoopSpan()


def span(name: str, **attributes: Any):
    """Open a span (context manager). No-op span when telemetry is disabled."""
    if not is_enabled():
        return _NOOP_SPAN
    try:
        return _logfire.span(name, **attributes)
    except Exception:
        _warn_once(f"span:{name}", f"telemetry span {name!r} failed")
        return _NOOP_SPAN


@contextmanager
def timed(name: str, metric: Optional["_Instrument"] = None, **attributes: Any):
    """Span + latency: times the block and records elapsed ms to `metric`.

    Always yields the span object so callers can set extra attributes. Records the
    duration metric even when spans are disabled (metrics may still be wanted).
    """
    start = time.perf_counter()
    sp = span(name, **attributes)
    try:
        with sp:
            yield sp
    finally:
        if metric is not None:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            metric.record(elapsed_ms, _metric_attrs(attributes))


def _metric_attrs(attributes: Dict[str, Any]) -> Dict[str, Any]:
    """Keep metric-label cardinality low: only string/bool attrs become labels.

    Numeric attrs (count, pool_size, n_candidates) stay on spans, not metric labels.
    """
    return {k: v for k, v in attributes.items() if isinstance(v, (str, bool))}


def set_attributes(sp, **attributes: Any) -> None:
    """Set attributes on a span if telemetry is on (helper to avoid call-site guards)."""
    if sp is None or sp is _NOOP_SPAN:
        return
    try:
        sp.set_attributes(attributes)
    except Exception:
        _warn_once("set_attributes", "telemetry set_attributes failed")


def instrument(*d_args, **d_kwargs):
    """Decorator: delegate to logfire.instrument (extract_args off by default for PII),
    or identity when disabled."""
    d_kwargs.setdefault("extract_args", False)

    def _wrap(fn):
        if not is_enabled():
            return fn
        try:
            return _logfire.instrument(*d_args, **d_kwargs)(fn)
        except Exception:
            _warn_once(f"instrument:{getattr(fn, '__name__', fn)}", "telemetry @instrument failed")
            return fn

    return _wrap


# ------------------------------------------------------------------- metrics
class _Instrument:
    """No-op metric instrument exposing the OTel-ish surface we call."""

    def add(self, *_a, **_k):
        pass

    def record(self, *_a, **_k):
        pass

    def set(self, *_a, **_k):
        pass


_NOOP_INSTRUMENT = _Instrument()
_METRIC_CACHE: Dict[str, Any] = {}


def counter(name: str, unit: str = "1", description: str = "") -> Any:
    return _get_metric("counter", name, unit, description)


def histogram(name: str, unit: str = "ms", description: str = "") -> Any:
    return _get_metric("histogram", name, unit, description)


def gauge(name: str, unit: str = "1", description: str = "") -> Any:
    return _get_metric("gauge", name, unit, description)


def gauge_callback(name: str, callback: Callable, unit: str = "1", description: str = "") -> None:
    """Register an observable gauge (e.g. cache size). No-op when disabled."""
    if not is_enabled():
        return
    try:
        _logfire.metric_gauge_callback(
            name, callbacks=[callback], unit=unit, description=description
        )
    except Exception:
        _warn_once(f"gauge_callback:{name}", f"telemetry gauge_callback {name!r} failed")


def _get_metric(kind: str, name: str, unit: str, description: str) -> Any:
    # Return (but do NOT cache) the no-op while disabled, so the real instrument
    # is created the first time it's fetched after configure().
    if not is_enabled():
        return _NOOP_INSTRUMENT
    key = f"{kind}:{name}"
    inst = _METRIC_CACHE.get(key)
    if inst is not None:
        return inst
    try:
        if kind == "counter":
            inst = _logfire.metric_counter(name, unit=unit, description=description)
        elif kind == "histogram":
            inst = _logfire.metric_histogram(name, unit=unit, description=description)
        else:
            inst = _logfire.metric_gauge(name, unit=unit, description=description)
    except Exception:
        _warn_once(f"metric:{key}", f"telemetry {kind} {name!r} creation failed")
        inst = _NOOP_INSTRUMENT
    _METRIC_CACHE[key] = inst
    return inst


# ----------------------------------------------------------- auto-instrument
def instrument_fastapi(app) -> None:
    if not is_enabled():
        return
    try:
        _logfire.instrument_fastapi(app, capture_headers=False)
    except Exception:
        logger.warning("logfire.instrument_fastapi() failed", exc_info=True)


def instrument_llm() -> None:
    """Instrument OpenAI + Anthropic clients (token usage auto-captured).

    Content capture follows `capture_content()`; the `include_content` kwarg name
    is version-dependent, so we try it and fall back to a bare call.
    """
    if not is_enabled():
        return
    for fn_name in ("instrument_openai", "instrument_anthropic"):
        fn = getattr(_logfire, fn_name, None)
        if fn is None:
            continue
        try:
            fn(include_content=_CAPTURE_CONTENT)
        except TypeError:
            try:
                fn()
            except Exception:
                # Usually just means that LLM SDK isn't installed — expected, quiet.
                logger.debug("logfire.%s() skipped", fn_name, exc_info=True)
        except Exception:
            logger.debug("logfire.%s() skipped", fn_name, exc_info=True)


def instrument_httpx() -> None:
    """Instrument httpx (covers the Turbopuffer HTTP client). Bodies/headers off (PII)."""
    if not is_enabled():
        return
    try:
        _logfire.instrument_httpx(capture_request_body=False, capture_response_body=False)
    except Exception:
        logger.warning("logfire.instrument_httpx() failed", exc_info=True)
