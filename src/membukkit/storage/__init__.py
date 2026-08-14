"""Pluggable storage backends for MEMBUKKIT memory."""

from __future__ import annotations

from typing import Optional

from membukkit.config import RetrievalConfig, StorageConfig
from membukkit.storage.base import (
    Candidate,
    CandidatePool,
    FactRecord,
    MemoryBackend,
    content_id,
)
from membukkit.storage.memory import InMemoryBackend
from membukkit.storage.localstore import LocalStore, list_stores, stores_root

__all__ = [
    "MemoryBackend",
    "FactRecord",
    "Candidate",
    "CandidatePool",
    "InMemoryBackend",
    "LocalStore",
    "list_stores",
    "stores_root",
    "content_id",
    "make_backend",
]


def make_backend(
    retrieval: RetrievalConfig,
    encoder,
    storage: Optional[StorageConfig] = None,
) -> MemoryBackend:
    """Construct the backend named by `storage` (defaults to in-memory)."""
    storage = storage or StorageConfig()
    kind = (storage.backend or "memory").lower()
    if kind in ("memory", "local", "inmemory", "in_memory"):
        return InMemoryBackend(retrieval, encoder)
    if kind in ("turbopuffer", "tpuf"):
        from membukkit.storage.turbopuffer import TurbopufferBackend

        return TurbopufferBackend(retrieval, encoder, storage)
    raise ValueError(f"unknown storage backend: {storage.backend!r}")
