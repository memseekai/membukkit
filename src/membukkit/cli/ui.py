"""`membukkit ui` — launch the local explainability GUI.

Serves the built React frontend (bundled in the package under ``ui_dist``)
from the same FastAPI process as the memory service, so the GUI is one
command with no Node required. For frontend development run Vite separately
(`cd ui && npm run dev`) — it proxies API calls to this server.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode


def _ui_dist() -> Path | None:
    packaged = Path(__file__).resolve().parents[1] / "ui_dist"
    if (packaged / "index.html").exists():
        return packaged
    checkout = Path(__file__).resolve().parents[3] / "ui" / "dist"
    if (checkout / "index.html").exists():
        return checkout
    return None


def ui_url(host: str, port: int, store: Optional[str] = None, tab: Optional[str] = None) -> str:
    """Build the browser URL, optionally deep-linking a store / tab."""
    base = f"http://{host}:{port}"
    params = {}
    if store:
        params["store"] = store
    if tab:
        params["tab"] = tab
    if not params:
        return base
    return f"{base}/?{urlencode(params)}"


def launch_ui(
    host: str,
    port: int,
    llm: str,
    store: Optional[str] = None,
    no_browser: bool = False,
    api_only: bool = False,
) -> None:
    """Start the local FastAPI + UI server (blocking)."""
    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit(
            "the GUI needs the service extra: pip install 'membukkit[service]'"
        ) from e

    from membukkit.service.local_app import create_local_app

    dist = _ui_dist()
    if dist is None and not api_only:
        print(
            "(frontend bundle not found — serving API only; build it with "
            "`cd ui && npm install && npm run build`)"
        )

    app = create_local_app(ui_dist=dist, llm=llm)
    url = ui_url(host, port, store=store, tab="ask" if store else None)
    print(f"MemBukkit UI: {url}", flush=True)
    print(
        "Create a store in the sidebar (or pass --demo …), then drop files on Ingest.",
        flush=True,
    )
    if not no_browser and dist is not None:
        webbrowser.open(url)
    uvicorn.run(app, host=host, port=port, log_level="warning")


def cmd_ui(args) -> None:
    store_name = None
    if getattr(args, "demo", None):
        from membukkit.cli.demo import ensure_demo_store

        try:
            store_name = ensure_demo_store(args.demo, llm=args.llm)
        except ValueError as e:
            raise SystemExit(str(e)) from e

    launch_ui(
        host=args.host,
        port=args.port,
        llm=args.llm,
        store=store_name,
        no_browser=args.no_browser,
        api_only=args.api_only,
    )


def register(sub) -> None:
    p = sub.add_parser("ui", help="Launch the local web UI (drag-and-drop + explainability)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8377)
    p.add_argument("--llm", default="openai:gpt-4o-mini")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--api-only", action="store_true")
    p.add_argument(
        "--demo",
        metavar="NAME",
        help="load a bundled demo store and open the UI with it selected",
    )
    p.set_defaults(func=cmd_ui)
