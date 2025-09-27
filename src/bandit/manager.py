"""Simple orchestrator helper around bandit implementations."""

from __future__ import annotations

import numpy as np

from ..types import BanditDecision
from .base import Bandit
from .utils import softmax


class BanditManager:
    """Wrapper that exposes a friendly decision object."""

    def __init__(self, policy: Bandit) -> None:
        self._policy = policy

    def select(self, scores: np.ndarray, phi: np.ndarray) -> BanditDecision:
        idx = self._policy.select(scores, phi)
        combined_scores = self._policy.last_scores
        probs = softmax(combined_scores, beta=self._policy.temperature)
        decision = BanditDecision(
            chosen_index=idx,
            propensities=probs.tolist(),
            scores=combined_scores.tolist(),
        )
        return decision

    def update(self, phi: np.ndarray, reward: float, chosen_idx: int | None = None) -> None:
        if chosen_idx is None:
            chosen_idx = self._policy.last_index
        self._policy.update(phi, reward, chosen_idx)
