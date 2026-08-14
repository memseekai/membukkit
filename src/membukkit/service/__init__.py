"""Multi-tenant memory service (FastAPI) over per-owner Turbopuffer namespaces."""

from __future__ import annotations

from membukkit.service.manager import MemoryService, ServiceConfig, namespace_for

__all__ = ["MemoryService", "ServiceConfig", "namespace_for", "create_app"]


def create_app(*args, **kwargs):
    from membukkit.service.app import create_app as _create_app

    return _create_app(*args, **kwargs)
