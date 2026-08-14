"""OpenAI embedding models behind the membukkit Encoder interface.

Spec syntax (used by ``--encoder``): ``openai:MODEL[@DIMS]``, e.g.
``openai:text-embedding-3-large@1536``. DIMS uses the API's native
Matryoshka truncation (``dimensions=``).

Embeddings are disk-cached (sqlite, sha256(text)-keyed, one cache file per
encoder spec) so interrupted or repeated runs never re-bill the API for the
same text.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import numpy as np


class OpenAIEncoder:
    """OpenAI embeddings with the ``encode(texts, normalize=)`` interface."""

    BATCH = 128
    MAX_CHARS = 24000  # first-pass clip; final limit is token-based
    MAX_TOKENS = 8000  # API rejects inputs > 8192 tokens

    def __init__(self, model: str, dims: int | None):
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model
        self._dims = dims
        self._tok = None

    def _clip(self, t: str) -> str:
        t = (t or " ")[: self.MAX_CHARS]
        # Char-based clipping assumes ~4 chars/token; dense code or unusual
        # unicode can hit 1-2 chars/token, so token-clip anything sizeable.
        if len(t) > 4000:
            if self._tok is None:
                import tiktoken

                self._tok = tiktoken.get_encoding("cl100k_base")
            ids = self._tok.encode(t, disallowed_special=())
            if len(ids) > self.MAX_TOKENS:
                t = self._tok.decode(ids[: self.MAX_TOKENS])
        return t

    def encode(self, texts, normalize: bool = True, show_progress: bool = False):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        items = [self._clip(t) for t in items]
        bar = None
        if show_progress and len(items) > self.BATCH:
            try:
                from tqdm.auto import tqdm

                bar = tqdm(total=len(items), desc="embed", unit="txt", leave=False)
            except Exception:
                bar = None
        vecs: list = []
        for i in range(0, len(items), self.BATCH):
            batch = items[i : i + self.BATCH]
            kwargs = {"model": self._model, "input": batch}
            if self._dims:
                kwargs["dimensions"] = self._dims
            for attempt in range(6):
                try:
                    resp = self._client.embeddings.create(**kwargs)
                    break
                except Exception as e:
                    if attempt == 5:
                        raise
                    time.sleep(2**attempt)
                    print(f"  embed retry {attempt + 1}: {e}", flush=True)
            vecs.extend(d.embedding for d in resp.data)
            if bar is not None:
                bar.update(len(batch))
        if bar is not None:
            bar.close()
        arr = np.asarray(vecs, dtype=np.float32)
        if normalize:
            arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        return arr[0] if single else arr


class CachedEncoder:
    """Sha256-keyed sqlite cache in front of any encoder with our interface."""

    def __init__(self, base, path: str):
        import sqlite3

        self._base = base
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")
        self._db.commit()

    def encode(self, texts, normalize: bool = True, show_progress: bool = False):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        keys = [hashlib.sha256(t.encode()).hexdigest() for t in items]
        out: list = [None] * len(items)
        missing = []
        for i, k in enumerate(keys):
            row = self._db.execute("SELECT v FROM emb WHERE k=?", (k,)).fetchone()
            if row:
                out[i] = np.frombuffer(row[0], dtype=np.float32)
            else:
                missing.append(i)
        if missing:
            vecs = np.asarray(
                self._base.encode(
                    [items[i] for i in missing],
                    normalize=normalize,
                    show_progress=show_progress,
                ),
                dtype=np.float32,
            )
            if vecs.ndim == 1:
                vecs = vecs.reshape(1, -1)
            for j, i in enumerate(missing):
                out[i] = vecs[j]
                self._db.execute(
                    "INSERT OR REPLACE INTO emb VALUES (?,?)", (keys[i], vecs[j].tobytes())
                )
            self._db.commit()
        arr = np.vstack(out)
        return arr[0] if single else arr


def make_openai_encoder(spec: str, cache_dir: str = ".membukkit_emb_cache"):
    """Build a disk-cached OpenAI encoder from an ``openai:MODEL[@DIMS]`` spec."""
    body = spec.split(":", 1)[1]
    model, _, dims = body.partition("@")
    base = OpenAIEncoder(model, int(dims) if dims else None)
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", spec).strip("_")
    return CachedEncoder(base, str(Path(cache_dir) / f"{slug}.sqlite"))
