"""Convenience wrapper providing persistence for bandit policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..types import BanditDecision
from .base import BanditPolicy, LinearBanditState


class BanditManager:
    """Wrap a bandit policy with optional state persistence."""

    def __init__(self, policy: BanditPolicy, state_path: Path | None = None) -> None:
        self._policy = policy
        self._state_path = state_path
        if self._state_path and self._state_path.exists():
            self._load_state()

    def select(self, features: Sequence[Sequence[float]]) -> BanditDecision:
        return self._policy.select_action(features)

    def update(self, chosen_index: int, reward: float, feature_vector: Sequence[float]) -> None:
        self._policy.update(chosen_index, reward, feature_vector)
        self._persist_state()

    def _persist_state(self) -> None:
        if not self._state_path:
            return
        state = self._policy.get_state().to_dict()
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with self._state_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle)

    def _load_state(self) -> None:
        if not self._state_path:
            return
        with self._state_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        state = LinearBanditState.from_dict(payload)
        self._policy.load_state(state)

