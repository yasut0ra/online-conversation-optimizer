"""Base interfaces for bandit policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from ..types import BanditDecision


@dataclass
class LinearBanditState:
    """Persistent state for linear bandit algorithms."""

    dim: int
    A: np.ndarray
    b: np.ndarray

    def to_dict(self) -> Dict[str, List[List[float]]]:
        return {
            "dim": self.dim,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, List[List[float]]]) -> "LinearBanditState":
        dim = int(payload["dim"])
        A = np.array(payload["A"], dtype=float)
        b = np.array(payload["b"], dtype=float)
        return cls(dim=dim, A=A, b=b)


class BanditPolicy(ABC):
    """Abstract interface for bandit strategies."""

    @abstractmethod
    def select_action(self, features: Sequence[Sequence[float]]) -> BanditDecision:
        """Return chosen action index, propensities, and raw scores."""

    @abstractmethod
    def update(
        self, chosen_index: int, reward: float, feature_vector: Sequence[float]
    ) -> None:
        """Update the policy state from an observed reward."""

    @abstractmethod
    def get_state(self) -> LinearBanditState:
        """Return the serialisable policy state."""

    @abstractmethod
    def load_state(self, state: LinearBanditState) -> None:
        """Restore policy state from persisted data."""

