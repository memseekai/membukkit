"""`membukkit demo` — bundled, one-command demo scenarios."""

from __future__ import annotations

import json
from pathlib import Path

# Packaged under src/membukkit/demos/ (ships in the wheel).
DEMOS_DIR = Path(__file__).resolve().parents[1] / "demos"


def available_demos() -> dict:
    """Demo name -> manifest dict, discovered from the demos/ directory."""
    out = {}
    if not DEMOS_DIR.exists():
        return out
    for d in sorted(DEMOS_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "demo.json"
        if manifest.exists():
            out[d.name] = json.loads(manifest.read_text())
    return out


def _store_fact_count(store_name: str) -> int | None:
    """Return fact count for an existing store, or None if the store is absent."""
    from membukkit.storage.localstore import LocalStore

    try:
        store = LocalStore(store_name, create=False)
    except FileNotFoundError:
        return None
    meta = store.meta()
    n = meta.get("n_facts")
    if isinstance(n, int):
        return n
    if store.facts_path.exists():
        with open(store.facts_path) as f:
            return sum(1 for line in f if line.strip())
    return 0


def ensure_demo_store(
    name: str,
    llm: str,
    prompt_pack: str | None = None,
    on_progress=None,
) -> str:
    """Validate demo name, ingest into ``demo-{name}`` if needed, return store name.

    Skips re-ingest when the target store already has facts. Does not run the
    canned demo questions — callers that want the Q&A loop use ``cmd_demo``.

    ``on_progress`` (optional ``Callable[[ProgressEvent], None]``) is used by
    the GUI stream path; when omitted, CLI ingest tqdm is used.
    """
    demos = available_demos()
    if name not in demos:
        names = ", ".join(demos) if demos else "(none found)"
        raise ValueError(
            f"unknown demo {name!r}; available: {names}\n"
            "try `membukkit demo --list`"
        )

    manifest = demos[name]
    store_name = f"demo-{name}"
    pack = prompt_pack or manifest.get("prompt_pack") or None
    n_existing = _store_fact_count(store_name)
    if n_existing is not None and n_existing > 0:
        print(f"using existing store {store_name} ({n_existing} facts)")
        # Keep pack/prompts aligned with the manifest even when skipping re-ingest
        # (older demo stores may have a wrong or missing pack).
        if pack:
            try:
                from membukkit.prompts.packs import load_prompt_pack
                from membukkit.storage.localstore import LocalStore

                store = LocalStore(store_name, create=False)
                cfg = load_prompt_pack(pack)
                store.update_meta(prompts=cfg.to_dict(), prompt_pack=pack)
            except Exception as e:
                print(f"(could not refresh prompt pack {pack!r}: {e})")
        return store_name

    demo_dir = DEMOS_DIR / name
    print(f"=== {manifest.get('title', name)} ===")
    print(manifest.get("description", ""), "\n")

    if on_progress is None:
        from membukkit.cli.commands import cmd_ingest

        class _Ns:
            pass

        ing = _Ns()
        ing.paths = [str(demo_dir / p) for p in manifest["data"]]
        ing.store = store_name
        ing.llm = llm
        ing.encoder = None
        ing.no_distill = manifest.get("no_distill", False)
        ing.prompt_pack = pack
        cmd_ingest(ing)
        return store_name

    # Stream path: ingest with the caller's progress callback (no CLI tqdm).
    from membukkit.cli.common import open_store
    from membukkit.ingest import parse_path

    paths = [demo_dir / p for p in manifest["data"]]
    docs = []
    for p in paths:
        docs.extend(parse_path(p))
    mem, store = open_store(
        store_name,
        llm=llm,
        distill=not manifest.get("no_distill", False),
        create=True,
        prompt_pack=pack,
    )
    if pack:
        store.update_meta(prompts=mem.prompts.to_dict(), prompt_pack=pack)
    for doc in docs:
        doc_id = store.add_document(
            doc.name, doc.sessions, doc.dates, doc_type=doc.doc_type, origin=doc.origin
        )
        mem.ingest(
            doc.sessions,
            dates=doc.dates,
            doc_id=doc_id,
            doc_name=doc.name,
            doc_type=doc.doc_type,
            on_progress=on_progress,
        )
    store.save_backend(mem.backend)
    return store_name


def cmd_demo(args) -> None:
    demos = available_demos()
    if args.list or not args.name:
        if not demos:
            print("no demos found — reinstall membukkit or check the package data")
            return
        print("available demos (run with `membukkit demo <name>`):\n")
        for name, m in demos.items():
            print(f"  {name:<20} {m.get('title', '')}")
            print(f"  {'':<20} {m.get('description', '')[:100]}")
        return

    pack = getattr(args, "prompt_pack", None)
    try:
        store_name = ensure_demo_store(args.name, llm=args.llm, prompt_pack=pack)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    manifest = demos[args.name]

    from membukkit.cli.commands import cmd_ask

    class _Ns:
        pass

    print("\n--- demo questions ---")
    demo_as_of = manifest.get("question_date") or None
    for q in manifest.get("questions", []):
        print(f"\nQ: {q}")
        ask = _Ns()
        ask.question = q
        ask.store = store_name
        ask.llm = args.llm
        ask.show_trace = args.show_trace
        ask.prompt_pack = pack
        ask.as_of = demo_as_of
        cmd_ask(ask)

    as_of_hint = f" --as-of {demo_as_of}" if demo_as_of else ""
    print(f"\ntry your own:  membukkit chat --store {store_name}{as_of_hint}")
    print(f"see the memory map:  membukkit buckets --store {store_name} --label")

    if getattr(args, "ui", False):
        from membukkit.cli.ui import launch_ui

        launch_ui(
            host=args.host,
            port=args.port,
            llm=args.llm,
            store=store_name,
            no_browser=args.no_browser,
        )


def register(sub) -> None:
    p = sub.add_parser("demo", help="Run a bundled demo scenario")
    p.add_argument("name", nargs="?", help="demo name (omit to list)")
    p.add_argument("--list", action="store_true", help="list available demos")
    p.add_argument("--llm", default="openai:gpt-4o-mini")
    p.add_argument("--show-trace", action="store_true",
                   help="show retrieval traces for each demo answer")
    p.add_argument("--ui", action="store_true",
                   help="open the GUI with this demo store after the Q&A")
    p.add_argument("--host", default="127.0.0.1", help="UI host (with --ui)")
    p.add_argument("--port", type=int, default=8377, help="UI port (with --ui)")
    p.add_argument("--no-browser", action="store_true",
                   help="print the UI URL but do not open a browser (with --ui)")
    p.add_argument(
        "--prompt-pack",
        default=None,
        help="use-case prompt pack for first-time ingest (e.g. personal_assistant)",
    )
    p.set_defaults(func=cmd_demo)
