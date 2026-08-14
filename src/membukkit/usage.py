"""Token usage metering and rough USD estimates for receipts."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Approximate $ per 1M *input* tokens (output priced the same for receipt simplicity).
PRICE_PER_M_INPUT: Dict[str, float] = {
    "gpt-4o-mini": 0.15,
    "gpt-4o": 2.50,
    "gpt-4.1-mini": 0.40,
    "gpt-4.1": 2.00,
    "o4-mini": 1.10,
    "claude-haiku": 0.80,
    "claude-sonnet": 3.00,
    "claude-opus": 15.00,
    "gemini-2.0-flash": 0.10,
    "gemini-2.5-flash": 0.15,
    "gemma": 0.0,
    "ollama": 0.0,
    "local": 0.0,
}

CONTEXT_WINDOW_DEFAULT = 128_000

# Warn when estimated distill input exceeds this (chars/4).
DISTILL_WARN_TOKENS = 2_000_000


@dataclass
class TokenUsage:
    """Accumulated prompt/completion tokens for one operation or store lifetime."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    source: str = "estimate"  # api | estimate | mixed
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)

    def add(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        *,
        source: str = "api",
        calls: int = 1,
    ) -> None:
        self.prompt_tokens += max(0, int(prompt_tokens))
        self.completion_tokens += max(0, int(completion_tokens))
        self.calls += max(0, int(calls))
        if self.source == source:
            return
        if self.source in ("", "estimate") and source == "api":
            self.source = "api" if self.prompt_tokens == prompt_tokens else "mixed"
        elif self.source == "api" and source == "estimate":
            self.source = "mixed"
        elif self.source != source:
            self.source = "mixed"

    def merge(self, other: Optional["TokenUsage"]) -> None:
        if other is None:
            return
        self.add(
            other.prompt_tokens,
            other.completion_tokens,
            source=other.source,
            calls=other.calls,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "source": self.source,
            "calls": self.calls,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TokenUsage":
        if not data:
            return cls()
        return cls(
            prompt_tokens=int(data.get("prompt_tokens") or 0),
            completion_tokens=int(data.get("completion_tokens") or 0),
            source=str(data.get("source") or "estimate"),
            calls=int(data.get("calls") or 0),
        )


def estimate_tokens(*parts: str) -> int:
    """Rough token estimate (chars/4) — same heuristic as the CLI."""
    return max(0, sum(len(p or "") for p in parts) // 4)


def price_per_m(model_spec: str) -> Optional[float]:
    """Return $/1M input tokens for a model spec, or None if unknown."""
    spec = (model_spec or "").lower()
    if not spec:
        return None
    if "ollama" in spec or spec.startswith("local:") or ":local" in spec:
        return 0.0
    for key, price in PRICE_PER_M_INPUT.items():
        if key in spec:
            return price
    return None


def estimate_cost_usd(
    usage: Optional[TokenUsage],
    model_spec: str = "",
) -> Optional[float]:
    """USD estimate from total tokens × input rate (rough receipt, not invoice)."""
    if usage is None or usage.total_tokens <= 0:
        return 0.0 if usage is not None else None
    rate = price_per_m(model_spec)
    if rate is None:
        return None
    return usage.total_tokens / 1e6 * rate


def window_fraction(
    tokens: int,
    context_window: int = CONTEXT_WINDOW_DEFAULT,
) -> float:
    if context_window <= 0:
        return 0.0
    return min(1.0, max(0.0, float(tokens) / float(context_window)))


def format_cost(usd: Optional[float]) -> str:
    if usd is None:
        return "cost n/a"
    if usd == 0.0:
        return "$0"
    if usd < 0.01:
        return f"${usd:.4f}"
    if usd < 1:
        return f"${usd:.3f}"
    return f"${usd:.2f}"


def format_usage_line(
    usage: Optional[TokenUsage],
    model_spec: str = "",
    *,
    reader_tokens: int = 0,
) -> str:
    """One-line human receipt fragment."""
    u = usage or TokenUsage()
    cost = estimate_cost_usd(u, model_spec)
    tag = "metered" if u.source == "api" else ("est." if u.source == "estimate" else "mixed")
    parts = [
        f"{u.prompt_tokens:,} in / {u.completion_tokens:,} out ({tag})",
    ]
    if cost is not None:
        parts.append(format_cost(cost))
    if reader_tokens > 0:
        pct = window_fraction(reader_tokens) * 100
        parts.append(f"~{reader_tokens:,} reader tok ({pct:.2f}% of 128k)")
    if model_spec:
        parts.append(model_spec)
    return " · ".join(parts)


class UsageMeter:
    """Accumulates TokenUsage; typically one per thread via get_meter()."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._usage = TokenUsage()

    def add(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        *,
        source: str = "api",
        calls: int = 1,
    ) -> None:
        with self._lock:
            self._usage.add(
                prompt_tokens, completion_tokens, source=source, calls=calls
            )

    def peek(self) -> TokenUsage:
        with self._lock:
            u = self._usage
            return TokenUsage(
                prompt_tokens=u.prompt_tokens,
                completion_tokens=u.completion_tokens,
                source=u.source,
                calls=u.calls,
            )

    def take(self) -> TokenUsage:
        with self._lock:
            out = self._usage
            self._usage = TokenUsage()
            return out


_thread_local = threading.local()


def get_meter() -> UsageMeter:
    m = getattr(_thread_local, "meter", None)
    if m is None:
        m = UsageMeter()
        _thread_local.meter = m
    return m


def record_api_usage(
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
) -> None:
    if not prompt_tokens and not completion_tokens:
        return
    get_meter().add(
        int(prompt_tokens or 0),
        int(completion_tokens or 0),
        source="api",
    )


def record_estimate_usage(prompt: str = "", completion: str = "") -> None:
    get_meter().add(
        estimate_tokens(prompt),
        estimate_tokens(completion),
        source="estimate",
    )


def merge_usage_into_meta(
    meta: Dict[str, Any],
    usage: Optional[TokenUsage],
    model_spec: str = "",
) -> Dict[str, Any]:
    """Return updated usage_totals dict for LocalStore meta."""
    prev = meta.get("usage_totals") if isinstance(meta.get("usage_totals"), dict) else {}
    totals = TokenUsage.from_dict(prev)
    totals.merge(usage)
    model = model_spec or str(prev.get("model") or "")
    cost = estimate_cost_usd(totals, model)
    out = totals.to_dict()
    out["est_cost_usd"] = cost if cost is not None else float(prev.get("est_cost_usd") or 0)
    out["model"] = model
    return out


def llm_model_spec(llm_fn: Any) -> str:
    """Best-effort model id from a backend instance."""
    if llm_fn is None:
        return ""
    model = getattr(llm_fn, "model", None)
    if model:
        name = type(llm_fn).__name__.replace("Backend", "").lower()
        if name and name != "local":
            return f"{name}:{model}"
        return str(model)
    return ""
