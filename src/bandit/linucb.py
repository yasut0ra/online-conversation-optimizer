"""LinUCB implementation with environment-configurable exploration."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .base import Bandit
from .utils import ensure_2d, get_env_float


class LinUCB(Bandit):
    """Linear UCB with shared parameter vector."""

    def __init__(
        self,
        alpha: Optional[float] = None,
        lam: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> None:
        super().__init__(beta=beta)
        self._alpha = (
            alpha
            if alpha is not None
            else get_env_float("LINUCB_ALPHA", 0.6)
        )
        self._lambda = (
            lam if lam is not None else get_env_float("BANDIT_LAMBDA", 1.0)
        )
        self._A: Optional[np.ndarray] = None
        self._b: Optional[np.ndarray] = None

    def _select_impl(
        self, prior_scores: np.ndarray, features: np.ndarray
    ) -> tuple[int, np.ndarray]:
        features = ensure_2d(features)
        dim = features.shape[1]
        self._ensure_state(dim)
        assert self._A is not None and self._b is not None

        A_inv = np.linalg.inv(self._A)
        theta = A_inv @ self._b
        means = features @ theta
        uncertainties = np.sqrt(np.sum(features @ A_inv * features, axis=1))
        scores = prior_scores + means + self._alpha * uncertainties
        chosen = int(np.argmax(scores))
        return chosen, scores

    def update(self, phi: np.ndarray, reward: float, chosen_idx: int) -> None:
        features = ensure_2d(phi)
        dim = features.shape[1]
        self._ensure_state(dim)
        assert self._A is not None and self._b is not None

        x = features[chosen_idx]
        self._A += np.outer(x, x)
        self._b += reward * x

    def _ensure_state(self, dim: int) -> None:
        if self._A is None or self._b is None:
            self._A = self._lambda * np.eye(dim)
            self._b = np.zeros(dim)
        elif self._A.shape[0] != dim:
            raise ValueError("Feature dimension mismatch for LinUCB")

