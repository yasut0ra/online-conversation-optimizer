"""Application configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    """Runtime configuration sourced from environment variables."""

    candidate_count: int = int(os.getenv("CANDIDATE_COUNT", "3"))
    log_path: Path = Path(os.getenv("LOG_PATH", "logs/interactions.jsonl"))
    bandit_state_path: Path = Path(os.getenv("BANDIT_STATE_PATH", "state/linucb.json"))
    bandit_policy: str = os.getenv("BANDIT_POLICY", "linucb")


def load_config() -> AppConfig:
    """Return the active application configuration."""

    return AppConfig()

