"""Linear Thompson Sampling bandit."""

from __future__ import annotations

import numpy as np

from .base import Bandit
from .utils import ensure_2d, get_env_float


class LinTS(Bandit):
    """Posterior sampling for linear contextual bandits."""

    def __init__(
        self,
        sigma2: float | None = None,
        lam: float | None = None,
        beta: float | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(beta=beta)
        self._sigma2 = (
            sigma2
            if sigma2 is not None
            else get_env_float("LINTS_SIGMA2", 0.5)
        )
        self._lambda = (
            lam if lam is not None else get_env_float("BANDIT_LAMBDA", 1.0)
        )
        self._rng = np.random.default_rng(random_state)
        self._A: np.ndarray | None = None
        self._b: np.ndarray | None = None

    def _select_impl(
        self, prior_scores: np.ndarray, features: np.ndarray
    ) -> tuple[int, np.ndarray]:
        features = ensure_2d(features)
        dim = features.shape[1]
        self._ensure_state(dim)
        assert self._A is not None and self._b is not None

        A_inv = np.linalg.inv(self._A)
        theta_bar = A_inv @ self._b
        cov = (self._sigma2) * A_inv
        theta_sample = self._rng.multivariate_normal(theta_bar, cov)
        scores = prior_scores + features @ theta_sample
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
            raise ValueError("Feature dimension mismatch for LinTS")

