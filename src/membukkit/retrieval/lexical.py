"""Optional BM25 lexical lane.

Dense routing is the shipped default and every published number comes from it.
This module backs `RetrievalConfig.lexical_lane`, an opt-in second retrieval
lane that finds facts by *term overlap* rather than embedding proximity, so
exact strings a bi-encoder under-weights (identifiers, error codes, filenames,
rare proper nouns) can reach the reader even when topic routing never opened
their bucket.

The lane adds candidates; it does not remove or reorder the routed ones. See
`InMemoryBackend.candidates` for the union and `MemorySystem._retrieve` for the
RRF fusion that consumes the scores.

Requires the `bm25` extra: ``pip install "membukkit[bm25]"``.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

_INSTALL_HINT = (
    "The BM25 lexical lane needs the `bm25` extra: pip install \"membukkit[bm25]\" "
    "(or set RetrievalConfig.lexical_lane=False to stay on dense routing only)."
)


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, underscores kept so snake_case survives.

    Deliberately plain: no stemming or stopword list, so a query term matches
    the stored term or it does not. That keeps the lane predictable, which is
    the point of having it next to a semantic retriever.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """BM25-Okapi over a fixed list of texts, addressed by list position."""

    def __init__(self, texts: Sequence[str]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - exercised via _require
            raise ImportError(_INSTALL_HINT) from exc

        self.n = len(texts)
        corpus = [tokenize(t) for t in texts]
        # rank_bm25 divides by the corpus average length; an all-empty corpus
        # would blow up, so keep a sentinel token for otherwise empty rows.
        self._bm25 = BM25Okapi([c or ["\x00"] for c in corpus])

    def top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Top-k `(position, score)` best-first. Zero-score hits are dropped:
        a document sharing no query term is not a lexical match."""
        toks = tokenize(query)
        if not toks or self.n == 0 or k <= 0:
            return []
        scores = self._bm25.get_scores(toks)
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        out: List[Tuple[int, float]] = []
        for i in order[:k]:
            s = float(scores[i])
            if s <= 0.0:
                break
            out.append((int(i), s))
        return out


def build_index(texts: Sequence[str]) -> Optional[BM25Index]:
    """Build an index, or None for an empty corpus."""
    if not texts:
        return None
    return BM25Index(texts)


def available() -> bool:
    """True when the `bm25` extra is installed."""
    try:
        import rank_bm25  # noqa: F401
    except ImportError:
        return False
    return True
