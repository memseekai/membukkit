"""Thin stdio MCP server: memory_add / memory_search / memory_ask.

Requires the optional ``mcp`` extra: ``pip install membukkit[mcp]``.
Supports the MCP Python SDK 1.x (FastMCP) and 2.x (MCPServer).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def _open_memory(store: str, llm: str):
    from membukkit.cli.common import open_store
    from membukkit.memory_api import Memory

    mem, local = open_store(store, llm=llm, create=True)
    return Memory.wrap(mem), local


def _make_server(name: str = "membukkit"):
    """Return (server, tool_decorator) for the installed mcp SDK."""
    try:
        from mcp.server.mcpserver import MCPServer

        server = MCPServer(name)
        return server, server.tool
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP

        server = FastMCP(name)
        return server, server.tool
    except ImportError as e:
        raise SystemExit(
            "MCP support requires the mcp package. Install with: pip install 'membukkit[mcp]'"
        ) from e


def build_mcp(store: str, llm: str):
    mcp, tool = _make_server("membukkit")

    @tool()
    def memory_add(
        text: str,
        date: Optional[str] = None,
        subject: str = "",
        store_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store a memory utterance. Returns a write receipt (n_stored, superseded, status)."""
        name = store_name or store
        mem, local = _open_memory(name, llm)
        report = mem.add(text, subject=subject or "", date=date)
        local.save_backend(mem.backend)
        if report.n_stored:
            local.update_meta(bucket_labels={})
        body = report.to_dict()
        body["store"] = name
        body["n_facts"] = mem.backend.count()
        return body

    @tool()
    def memory_search(
        query: str,
        as_of: Optional[str] = None,
        top_k: int = 10,
        store_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve dated memory evidence without generating an answer."""
        name = store_name or store
        mem, _ = _open_memory(name, llm)
        if mem.backend.count() == 0:
            return {"query": query, "hits": [], "error": "store is empty"}
        res = mem.search(query, as_of=as_of, top_k=top_k, include_history=True)
        return {
            "query": res.query,
            "store": name,
            "hits": [
                {
                    "fact": h.fact,
                    "text": h.text,
                    "timestamp": h.timestamp,
                    "status": h.status,
                    "source_ref": h.source_ref,
                    "doc_name": h.doc_name,
                }
                for h in res.hits
            ],
            "est_reader_tokens": getattr(res.trace, "est_reader_tokens", 0),
            "n_scanned": res.trace.n_scanned,
            "n_facts": res.trace.n_facts,
            "usage": getattr(res.trace, "usage", None),
            "est_cost_usd": getattr(res.trace, "est_cost_usd", None),
            "window_fraction": getattr(res.trace, "window_fraction", None),
        }

    @tool()
    def memory_ask(
        query: str,
        as_of: Optional[str] = None,
        store_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Answer a question from dated memory with evidence receipts."""
        name = store_name or store
        mem, _ = _open_memory(name, llm)
        if mem.backend.count() == 0:
            return {"answer": None, "error": "store is empty", "store": name}
        receipt = mem.ask(query, as_of=as_of, include_history=True)
        return {
            "answer": receipt.answer,
            "store": name,
            "question_date": receipt.question_date,
            "est_reader_tokens": receipt.est_reader_tokens,
            "scan_fraction": receipt.scan_fraction,
            "n_scanned": receipt.n_scanned,
            "n_facts": receipt.n_facts,
            "reader_type": receipt.reader_type,
            "usage": receipt.usage,
            "est_cost_usd": receipt.est_cost_usd,
            "window_fraction": receipt.window_fraction,
            "model": receipt.model,
            "evidence": [
                {
                    "fact": e.fact,
                    "status": e.status,
                    "timestamp": e.timestamp,
                    "source_ref": e.source_ref,
                }
                for e in receipt.evidence[:20]
            ],
        }

    return mcp


def run_mcp(
    store: Optional[str] = None,
    llm: str = "openai:gpt-4o-mini",
) -> None:
    """Run the MCP server on stdio (for Cursor / Claude Desktop)."""
    name = (store or os.environ.get("MEMBUKKIT_STORE") or "default").strip()
    mcp = build_mcp(name, llm)
    mcp.run(transport="stdio")


def tool_names() -> list:
    """Return registered tool names (builds server; does not start stdio)."""
    build_mcp("default", "openai:gpt-4o-mini")
    return ["memory_add", "memory_search", "memory_ask"]


def dump_tool_catalog() -> str:
    return json.dumps({"tools": tool_names()}, indent=2)
