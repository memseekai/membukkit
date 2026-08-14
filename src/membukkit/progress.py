"""Shared progress events for ingest, distill, label, embed, and eval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class ProgressEvent:
    """One progress tick.

    ``phase`` is a short machine token: parse | distill | embed | label |
    retrieve | answer | judge | done | error.
    ``total`` may be 0 while the caller is still sizing the work.
    """

    phase: str
    done: int
    total: int
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


ProgressCallback = Optional[Callable[[ProgressEvent], None]]


def emit(
    on_progress: ProgressCallback,
    phase: str,
    done: int,
    total: int,
    detail: str = "",
) -> None:
    """Fire a progress callback if one is installed."""
    if on_progress is None:
        return
    on_progress(ProgressEvent(phase=phase, done=done, total=total, detail=detail))


_EMBED_BATCH = 64


class ProgressFileWriter:
    """Throttled writer for ``{output_dir}/progress.json`` (bench / GUI poll)."""

    def __init__(self, path, min_interval_s: float = 0.5):
        from pathlib import Path

        self.path = Path(path)
        self.min_interval_s = min_interval_s
        self._last = 0.0
        self._last_phase = None

    def write(self, phase: str, done: int, total: int, detail: str = "", *, force: bool = False) -> None:
        import json
        import time
        from datetime import datetime, timezone

        now = time.monotonic()
        phase_changed = phase != self._last_phase
        if not force and not phase_changed and (now - self._last) < self.min_interval_s:
            return
        self._last = now
        self._last_phase = phase
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "phase": phase,
            "done": int(done),
            "total": int(total),
            "detail": detail or "",
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(self.path)

    def callback(self) -> Callable[[ProgressEvent], None]:
        def _cb(ev: ProgressEvent) -> None:
            self.write(ev.phase, ev.done, ev.total, ev.detail)

        return _cb


def encode_with_progress(encoder, texts, *, on_progress: ProgressCallback = None, normalize: bool = True):
    """Encode texts, emitting ``embed`` ProgressEvents when a callback is set.

    Batches so long embeds update the bar; without a callback this is a single
    ``encoder.encode`` call (same as before).
    """
    import numpy as np

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    if on_progress is None:
        return encoder.encode(texts, normalize=normalize)

    emit(on_progress, "embed", 0, len(texts), detail="embedding")
    parts = []
    for i in range(0, len(texts), _EMBED_BATCH):
        batch = texts[i : i + _EMBED_BATCH]
        vecs = np.asarray(
            encoder.encode(batch, normalize=normalize, show_progress=False),
            dtype=np.float32,
        )
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        parts.append(vecs)
        done = min(i + len(batch), len(texts))
        emit(on_progress, "embed", done, len(texts), detail=f"embedded {done}/{len(texts)}")
    return np.vstack(parts)
