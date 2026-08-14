#!/usr/bin/env python3
"""Rewrite README image paths for a PyPI upload, and back again.

GitHub resolves repo-relative image paths in both private and public repos, so
that is what README.md holds. PyPI has no repo context and would render those
as broken images, so the long_description needs absolute URLs.

Absolute `raw.githubusercontent.com` URLs only resolve once the repo is public.
Run this immediately before building for PyPI, and revert straight after:

    python scripts/pypi_readme.py --absolute
    uv build
    python scripts/pypi_readme.py --relative

`--check` reports the current state without touching anything.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RAW = "https://raw.githubusercontent.com/memseekai/membukkit/main/"
README = Path(__file__).resolve().parents[1] / "README.md"


def _local_images(text: str) -> list[str]:
    return [
        u
        for u in re.findall(r'<img src="([^"]+)"', text)
        if not u.startswith("http")
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--absolute", action="store_true", help="relative -> raw URLs (pre-build)")
    g.add_argument("--relative", action="store_true", help="raw URLs -> relative (post-build)")
    g.add_argument("--check", action="store_true", help="report state, change nothing")
    args = ap.parse_args()

    text = README.read_text()
    local = _local_images(text)
    absolute = text.count(f'<img src="{RAW}')

    if args.check:
        print(f"{len(local)} relative image(s), {absolute} absolute image(s)")
        print("state:", "PyPI-ready" if not local else "GitHub-ready")
        return 0

    if args.absolute:
        out = re.sub(r'<img src="(?!https?://)([^"]+)"', lambda m: f'<img src="{RAW}{m.group(1)}"', text)
        verb = "absolute"
    else:
        out = text.replace(f'<img src="{RAW}', '<img src="')
        verb = "relative"

    if out == text:
        print(f"already {verb}; nothing to do")
        return 0
    README.write_text(out)
    print(f"rewrote {README.name} image paths -> {verb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
