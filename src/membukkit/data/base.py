"""Shared dataset interfaces for MEMBUKKIT."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FactInput:
    """A fact to be stored in memory."""

    text: str
    timestamp: datetime
    tag: str = "NEW_OBS"
    source_session: Optional[str] = None
    source_speaker: Optional[str] = None


@dataclass
class QueryInput:
    """A query to be evaluated against memory."""

    text: str
    query_type: Optional[str] = None
    ground_truth: Optional[str] = None
    evidence: Optional[List[str]] = None
    category: Optional[int] = None


class CoreMemDataset(ABC):
    """Abstract dataset interface."""

    @abstractmethod
    def get_facts(self) -> List[FactInput]:
        ...

    @abstractmethod
    def get_queries(self) -> List[QueryInput]:
        ...

    @abstractmethod
    def get_utility_matrix(self) -> np.ndarray:
        ...

    @abstractmethod
    def evaluate(self, predictions: Any) -> Dict[str, float]:
        ...
