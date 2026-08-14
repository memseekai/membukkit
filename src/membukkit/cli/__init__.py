"""MemBukkit CLI.

Human-facing commands (ingest / add / ask / chat / search / buckets / stores /
demo / bench / ui / mcp) live here; the research-grade benchmark harness keeps its
full flag surface under ``membukkit eval`` / ``rag-eval`` / ``train-*`` /
``serve`` (see ``eval_legacy``).
"""

from __future__ import annotations

import argparse
import logging
import sys

_LEGACY_COMMANDS = {"eval", "rag-eval", "train-encoder", "train-reranker", "serve"}


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    from membukkit.credentials import bootstrap_credentials

    bootstrap_credentials()

    if argv and argv[0] in _LEGACY_COMMANDS:
        from membukkit.cli import eval_legacy

        sys.argv = [sys.argv[0]] + argv
        eval_legacy.main()
        return

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "google_genai", "google"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        prog="membukkit",
        description="MemBukkit: explainable long-term memory for LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "quickstart:\n"
            "  membukkit ui --demo personal-assistant   # GUI on a preloaded demo store\n"
            "  # bring an LLM: paste a key in the GUI, export OPENAI_API_KEY=sk-...,\n"
            "  # or stay local with --llm ollama:llama3.1\n"
            "  membukkit add \"rent is 800€\" --store notes --date 2024-01-10\n"
            "  membukkit ask --store notes \"How much is rent?\" --as-of 2024-05-01\n"
            "\n"
            "note: `ui` = local GUI over disk stores; `serve` = multi-tenant HTTP service "
            "(advanced)."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    from membukkit.cli import commands

    commands.register(sub)

    from membukkit.cli import bench

    bench.register(sub)

    from membukkit.cli import demo

    demo.register(sub)

    from membukkit.cli import ui

    ui.register(sub)

    from membukkit.cli import mcp_cmd

    mcp_cmd.register(sub)

    # Discoverability stubs for the research / advanced commands (dispatched above).
    for name, help_text in (
        ("eval", "[advanced] Run a benchmark evaluation"),
        ("rag-eval", "[advanced] Run multi-hop RAG benchmarks"),
        ("train-encoder", "[advanced] Fine-tune the bi-encoder"),
        ("train-reranker", "[advanced] Fine-tune the cross-encoder reranker"),
        ("serve", "[advanced] Multi-tenant HTTP service (not the local GUI; see `ui`)"),
    ):
        sub.add_parser(name, help=help_text, add_help=False)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0 if not args.command else 2)
    args.func(args)
