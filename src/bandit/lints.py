"""Linear Thompson Sampling (LinTS) contextual bandit."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..types import BanditDecision
from .base import BanditPolicy, LinearBanditState
from .linucb import _softmax


class LinTSPolicy(BanditPolicy):
    """Posterior sampling for linear bandits."""

    def __init__(
        self,
        exploration_variance: float = 0.5,
        regularization: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self._v = exploration_variance
        self._lambda = regularization
        self._rng = np.random.default_rng(random_state)
        self._state: LinearBanditState | None = None

    def select_action(self, features: Sequence[Sequence[float]]) -> BanditDecision:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Features must be 2D: actions x dimension")

        self._ensure_state(matrix.shape[1])
        assert self._state is not None
        invA = np.linalg.inv(self._state.A)
        theta_bar = invA @ self._state.b
        cov = (self._v ** 2) * invA
        theta_sample = self._rng.multivariate_normal(theta_bar, cov)

        scores = matrix @ theta_sample
        propensities = _softmax(scores)
        chosen_index = int(np.argmax(scores))
        return BanditDecision(
            chosen_index=chosen_index,
            propensities=propensities.tolist(),
            scores=scores.tolist(),
        )

    def update(
        self, chosen_index: int, reward: float, feature_vector: Sequence[float]
    ) -> None:
        vec = np.asarray(feature_vector, dtype=float)
        self._ensure_state(vec.shape[0])
        assert self._state is not None
        self._state.A += np.outer(vec, vec)
        self._state.b += reward * vec

    def get_state(self) -> LinearBanditState:
        if self._state is None:
            raise RuntimeError("State not initialised")
        return self._state

    def load_state(self, state: LinearBanditState) -> None:
        self._state = state

    def _ensure_state(self, dim: int) -> None:
        if self._state is None:
            self._state = LinearBanditState(
                dim=dim,
                A=self._lambda * np.eye(dim),
                b=np.zeros(dim),
            )
        elif self._state.dim != dim:
            raise ValueError(
                f"Feature dimension {dim} mismatched with existing {self._state.dim}"
            )

