"""CLI: membukkit mcp — thin stdio MCP server."""

from __future__ import annotations

import os
import sys

from membukkit.cli.common import DEFAULT_LLM


def cmd_mcp(args) -> None:
    from membukkit import mcp_server

    if getattr(args, "list_tools", False):
        print(mcp_server.dump_tool_catalog())
        return
    store = args.store or os.environ.get("MEMBUKKIT_STORE") or "default"
    print(
        f"MCP server on stdio for store {store!r} — leave this running; "
        "your MCP client connects here (Ctrl-C to stop).",
        file=sys.stderr,
    )
    mcp_server.run_mcp(store=store, llm=args.llm)


def register(sub) -> None:
    p = sub.add_parser(
        "mcp",
        help="Run MCP server over stdio (blocks until client disconnects / Ctrl-C)",
    )
    p.add_argument(
        "--store",
        default=None,
        help="default store name (or set MEMBUKKIT_STORE)",
    )
    p.add_argument("--llm", default=DEFAULT_LLM)
    p.add_argument(
        "--list-tools",
        action="store_true",
        help="print tool catalog as JSON and exit (requires mcp extra)",
    )
    p.set_defaults(func=cmd_mcp)
