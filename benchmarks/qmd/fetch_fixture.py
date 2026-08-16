"""Re-vendor QMD's benchmark fixture at a pinned commit.

    python -m benchmarks.qmd.fetch_fixture              # pin to current upstream HEAD
    python -m benchmarks.qmd.fetch_fixture --ref <sha>  # pin to a specific commit

Downloads QMD's ``src/bench/fixtures/example.json`` and the six markdown
documents in ``test/eval-docs/``, then writes ``MANIFEST.json`` recording the
upstream repo, the exact commit, and a sha256 for every vendored file so the
corpus is reproducible and tamper-evident.

Needs no auth: the QMD repository is public.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import urllib.request

REPO = "tobi/qmd"
FIXTURE_PATH = "src/bench/fixtures/example.json"
DOCS_PATH = "test/eval-docs"
OUT = pathlib.Path(__file__).parent / "fixture"
_API = "https://api.github.com"
_RAW = "https://raw.githubusercontent.com"


def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "membukkit-bench"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _get_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "membukkit-bench"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch(ref: str | None = None) -> str:
    sha = ref or _get_json(f"{_API}/repos/{REPO}/commits/main")["sha"]
    docs_out = OUT / "eval-docs"
    docs_out.mkdir(parents=True, exist_ok=True)

    (OUT / "example.json").write_bytes(_get_bytes(f"{_RAW}/{REPO}/{sha}/{FIXTURE_PATH}"))
    listing = _get_json(f"{_API}/repos/{REPO}/contents/{DOCS_PATH}?ref={sha}")
    for entry in listing:
        if entry["type"] == "file":
            (docs_out / entry["name"]).write_bytes(
                _get_bytes(f"{_RAW}/{REPO}/{sha}/{DOCS_PATH}/{entry['name']}")
            )

    checksums = {
        str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(OUT.rglob("*"))
        if p.is_file() and p.name != "MANIFEST.json"
    }
    (OUT / "MANIFEST.json").write_text(
        json.dumps(
            {
                "upstream_repo": f"https://github.com/{REPO}",
                "upstream_commit": sha,
                "fetched_paths": {"example.json": FIXTURE_PATH, "eval-docs/": f"{DOCS_PATH}/"},
                "note": "Vendored verbatim. Do not edit; re-run this script to update.",
                "sha256": checksums,
            },
            indent=2,
        )
        + "\n"
    )
    return sha


def verify() -> bool:
    """True when every vendored file still matches the manifest checksum."""
    manifest = json.loads((OUT / "MANIFEST.json").read_text())
    ok = True
    for rel, expected in manifest["sha256"].items():
        actual = hashlib.sha256((OUT / rel).read_bytes()).hexdigest()
        if actual != expected:
            print(f"  CHANGED {rel}")
            ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default=None, help="commit SHA to pin (default: upstream HEAD)")
    ap.add_argument("--verify", action="store_true", help="check checksums, fetch nothing")
    args = ap.parse_args()

    if args.verify:
        ok = verify()
        print("fixture matches manifest" if ok else "fixture DIFFERS from manifest")
        return 0 if ok else 1

    sha = fetch(args.ref)
    print(f"vendored QMD fixture at {sha}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
