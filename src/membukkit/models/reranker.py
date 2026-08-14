"""Cross-encoder reranker for utility-based fact scoring."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

DEFAULT_BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class ScoredFact:
    fact_idx: int
    text: str
    utility: float
    rank: int = -1
    cosine: float = 0.0
    entities: List[str] = field(default_factory=list)
    time_bucket: str = ""
    topic_bucket: int = -1

    def to_dict(self) -> Dict:
        return asdict(self)


class UtilityReranker:
    def __init__(self, base_model: str = DEFAULT_BASE, max_length: int = 256, device: Optional[str] = None):
        from sentence_transformers import CrossEncoder
        import torch

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.max_length = max_length
        self.base_model = base_model
        self.model = CrossEncoder(base_model, num_labels=1, max_length=max_length, device=device)

    def train(
        self,
        train_pairs,
        epochs: int = 2,
        batch_size: int = 32,
        lr: float = 2e-5,
        warmup_frac: float = 0.1,
        output_path: Optional[str] = None,
    ):
        from sentence_transformers import InputExample
        from torch.utils.data import DataLoader

        examples = [InputExample(texts=[q, f], label=float(lbl)) for q, f, lbl in train_pairs]
        # A plain list is a valid map-style dataset at runtime; torch's stubs only
        # accept Dataset, so the type here is over-strict.
        loader = DataLoader(examples, shuffle=True, batch_size=batch_size)  # ty: ignore[invalid-argument-type]
        warmup = int(len(loader) * epochs * warmup_frac)
        self.model.fit(
            train_dataloader=loader,
            epochs=epochs,
            warmup_steps=warmup,
            optimizer_params={"lr": lr},
            output_path=output_path,
            show_progress_bar=True,
        )

    def score(self, query: str, fact_texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not fact_texts:
            return np.zeros(0, dtype=np.float32)
        pairs = [(query, f) for f in fact_texts]
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32).reshape(-1)

    def explain(
        self,
        query: str,
        fact_texts: List[str],
        candidate_indices: List[int],
        top_k: int = 10,
        cosine_scores: Optional[np.ndarray] = None,
        index=None,
    ) -> List[ScoredFact]:
        utilities = self.score(query, fact_texts)
        order = np.argsort(utilities)[::-1]
        scored = []
        for rank, j in enumerate(order[:top_k]):
            fidx = candidate_indices[j]
            sf = ScoredFact(
                fact_idx=int(fidx),
                text=fact_texts[j][:200],
                utility=float(utilities[j]),
                rank=rank,
                cosine=float(cosine_scores[j]) if cosine_scores is not None else 0.0,
            )
            scored.append(sf)
        return scored

    def save(self, path: str):
        self.model.save(path)

    @classmethod
    def load(cls, path: str, max_length: int = 256, device: Optional[str] = None) -> "UtilityReranker":
        obj = cls.__new__(cls)
        from sentence_transformers import CrossEncoder
        import torch

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu")
        obj.device = device
        obj.max_length = max_length
        obj.base_model = path
        obj.model = CrossEncoder(path, num_labels=1, max_length=max_length, device=device)
        return obj
