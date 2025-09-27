"""Abstract base class for contextual bandits."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .utils import ensure_1d, ensure_2d, get_env_float, softmax


class Bandit(ABC):
    """Minimal bandit interface shared across algorithms."""

    def __init__(self, beta: Optional[float] = None) -> None:
        self._beta = beta if beta is not None else get_env_float("SOFTMAX_BETA", 1.0)
        self._last_scores: Optional[np.ndarray] = None
        self._last_idx: Optional[int] = None

    def select(self, scores: np.ndarray, phi: np.ndarray) -> int:
        """Pick an action index given prior scores and feature matrix."""

        prior = ensure_1d(scores)
        features = ensure_2d(phi)
        if prior.shape[0] != features.shape[0]:
            raise ValueError("scores and feature matrix must align on action axis")

        idx, combined_scores = self._select_impl(prior, features)
        self._last_scores = combined_scores
        self._last_idx = idx
        return idx

    @abstractmethod
    def _select_impl(
        self, prior_scores: np.ndarray, features: np.ndarray
    ) -> tuple[int, np.ndarray]:
        """Return chosen index and the scores used for action selection."""

    @abstractmethod
    def update(self, phi: np.ndarray, reward: float, chosen_idx: int) -> None:
        """Update model parameters using full feature matrix and observed reward."""

    def propensity(self, scores: Optional[np.ndarray] = None) -> float:
        """Return the softmax probability of the last chosen action."""

        if scores is None:
            if self._last_scores is None or self._last_idx is None:
                raise RuntimeError("No selection made yet")
            scores = self._last_scores
        else:
            scores = ensure_1d(scores)
            self._last_scores = scores
        if self._last_idx is None:
            raise RuntimeError("No action index available for propensity")
        probs = softmax(scores, beta=self._beta)
        return float(probs[self._last_idx])

    @property
    def last_scores(self) -> np.ndarray:
        if self._last_scores is None:
            raise RuntimeError("No scores cached; call select first")
        return self._last_scores

    @property
    def last_index(self) -> int:
        if self._last_idx is None:
            raise RuntimeError("No index cached; call select first")
        return self._last_idx

    @property
    def temperature(self) -> float:
        return self._beta
