"""Bi-encoder wrapper with lazy loading."""
from __future__ import annotations


class Encoder:
    def __init__(self, path: str):
        self._path = path
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._path)
        return self._model

    def encode(self, texts, normalize: bool = True, show_progress: bool = False):
        import numpy as np
        return np.asarray(self.model.encode(
            texts, show_progress_bar=show_progress, normalize_embeddings=normalize
        ))
