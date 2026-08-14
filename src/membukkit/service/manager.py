"""MemoryService — multi-tenant orchestration over per-owner namespaces.

The expensive models (bi-encoder + cross-encoder) are loaded ONCE and shared
across all tenants; each tenant gets a lightweight `MemorySystem` bound to its
own Turbopuffer namespace (`<prefix><owner>`, prefix `mem_` by default; set
`namespace_prefix=""` to address a namespace by owner id verbatim). `MemorySystem`
instances are cached (LRU) per worker; the underlying storage is stateless, so
any worker can serve any tenant.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional

from membukkit import telemetry
from membukkit.config import ModelConfig, PromptConfig, RetrievalConfig, StorageConfig

logger = logging.getLogger(__name__)

_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")


def namespace_for(owner: str, prefix: str = "mem_") -> str:
    """Map an owner id to a safe Turbopuffer namespace.

    ``prefix`` is prepended to the sanitized owner (default ``mem_``). Pass
    ``prefix=""`` to address a namespace by owner id verbatim (e.g. to point the
    service at a pre-existing, bespoke namespace).
    """
    clean = _SAFE.sub("_", (owner or "").strip())[:48].strip("_")
    if not clean:
        raise ValueError("owner id is empty or invalid")
    return f"{prefix}{clean}"


@dataclass
class ServiceConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig.default)
    llm: str = "openai:gpt-4o-mini"
    region: Optional[str] = None
    api_key: Optional[str] = None
    vector_dtype: str = "f16"
    namespace_prefix: str = "mem_"  # prepended to owner id; "" addresses a namespace verbatim
    cache_size: int = 256  # max cached per-tenant MemorySystems per worker
    # --- observability ---
    telemetry: bool = True  # configure Logfire/OTel on startup
    environment: Optional[str] = None
    capture_content: bool = False  # log raw fact/query/LLM text (PII; debug only)


class MemoryService:
    """Holds shared models and hands out per-tenant MemorySystems."""

    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig()
        self._lock = threading.Lock()
        self._cache: "OrderedDict[str, object]" = OrderedDict()
        self._encoder = None
        self._reranker = None
        self._llm_fn = None
        self.config.region = self.config.region or os.environ.get("TURBOPUFFER_REGION")
        self.config.api_key = self.config.api_key or os.environ.get("TURBOPUFFER_API_KEY")

    # ---------------------------------------------------------- shared models
    def _ensure_models(self):
        if self._encoder is not None:
            return
        with self._lock:
            if self._encoder is not None:
                return
            from membukkit.models.registry import resolve_encoder_path, resolve_reranker_path
            from membukkit.models.encoder import Encoder
            from membukkit.models.reranker import UtilityReranker
            from membukkit.llm.backends import parse_llm_spec

            m = self.config.models
            logger.info("loading shared encoder + reranker (one-time)")
            self._encoder = Encoder(resolve_encoder_path(m))
            self._reranker = UtilityReranker.load(resolve_reranker_path(m), device=m.device)
            self._llm_fn = parse_llm_spec(self.config.llm)

    # -------------------------------------------------------- per-tenant system
    def get(self, owner: str):
        """Return (constructing/caching as needed) the MemorySystem for an owner."""
        self._ensure_models()
        ns = namespace_for(owner, self.config.namespace_prefix)
        with self._lock:
            sys = self._cache.get(ns)
            if sys is not None:
                self._cache.move_to_end(ns)
                return sys

        from membukkit.pipeline import MemorySystem
        from membukkit.storage.turbopuffer import TurbopufferBackend
        from membukkit.extraction.distiller import FactDistiller

        storage = StorageConfig(
            backend="turbopuffer",
            namespace=ns,
            region=self.config.region,
            api_key=self.config.api_key,
            vector_dtype=self.config.vector_dtype,
        )
        backend = TurbopufferBackend(self.config.retrieval, self._encoder, storage)
        assert self._llm_fn is not None
        sys = MemorySystem(
            encoder=self._encoder,
            reranker=self._reranker,
            llm_fn=self._llm_fn,
            retrieval=self.config.retrieval,
            prompts=self.config.prompts,
            distiller=FactDistiller(self._llm_fn),
            backend=backend,
        )
        with self._lock:
            self._cache[ns] = sys
            self._cache.move_to_end(ns)
            while len(self._cache) > self.config.cache_size:
                self._cache.popitem(last=False)
        return sys

    def register_metrics(self) -> None:
        """Register an observable gauge for the per-worker tenant-cache size."""

        def _cached(_options):
            from opentelemetry.metrics import Observation

            return [Observation(len(self._cache))]

        telemetry.gauge_callback("membukkit.tenants.cached", _cached)

    def warm(self, owner: str) -> None:
        """Best-effort: prefetch centroids + warm the namespace cache (SSD/RAM).

        Call on session-open so the first real query doesn't pay the cold
        object-storage fetch.
        """
        try:
            sys = self.get(owner)
            sys._backend.partition()  # loads/caches centroids; touches the namespace
        except Exception as e:
            logger.debug("warm(%s) failed: %s", owner, e)

    def recluster(self, owner: str) -> bool:
        """Run the background re-cluster for one tenant if growth warrants it."""
        sys = self.get(owner)
        return sys._backend.maybe_recluster()

    def delete(self, owner: str) -> None:
        sys = self.get(owner)
        sys._backend.delete()
        with self._lock:
            self._cache.pop(namespace_for(owner, self.config.namespace_prefix), None)
