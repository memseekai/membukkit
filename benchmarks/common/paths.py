"""Filename derivation shared by the corpus exporters.

Exported filenames are the identity a competing retriever reports and the
identity gold labels are matched on, so getting them wrong silently corrupts a
comparison rather than failing loudly.
"""

from __future__ import annotations

import re
from typing import Dict

_UNSAFE = re.compile(r"[^a-zA-Z0-9._-]+")
_MAX_LEN = 120


def slugify(title: str) -> str:
    """Filesystem-safe stem for a document title.

    Leading dots are stripped deliberately. Wikipedia titles like ``.hack//Sign``
    and ``.50 BMG`` would otherwise produce dotfiles, which most indexers skip
    by default. That would hand one system a smaller corpus than the other and
    quietly invalidate the comparison, so it is prevented here rather than
    discovered later in a results diff.
    """
    slug = _UNSAFE.sub("-", title).strip("-").lower().lstrip(".")
    return (slug or "untitled")[:_MAX_LEN]


def unique_filename(title: str, used: Dict[str, int], suffix: str = ".md") -> str:
    """A collision-free filename for ``title``, mutating ``used`` as a ledger.

    Disambiguation retries rather than trusting a single counter: a generated
    ``foo-1`` can itself collide with a real title that slugifies to ``foo-1``.
    """
    base = slugify(title)
    n = used.get(base, 0)
    while True:
        candidate = f"{base}{suffix}" if n == 0 else f"{base}-{n}{suffix}"
        if candidate not in used:
            used[base] = n + 1
            used[candidate] = 1  # reserve the concrete filename too
            return candidate
        n += 1
