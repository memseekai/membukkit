"""Thin MCP server registration (optional mcp extra)."""

from __future__ import annotations

import pytest

mcp = pytest.importorskip("mcp")


def test_mcp_tool_catalog():
    from membukkit.mcp_server import tool_names

    names = tool_names()
    assert names == ["memory_add", "memory_search", "memory_ask"]
