"""Application configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _ensure_env_loaded() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _split_list(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


@dataclass
class AppConfig:
    """Runtime configuration sourced from environment variables."""

    openai_api_key: str | None
    candidate_count: int
    styles_whitelist: list[str] | None
    log_path: Path = field(default_factory=lambda: Path("logs/interactions.jsonl"))
    bandit_algo: str = "linucb"


def load_config() -> AppConfig:
    """Return the active application configuration."""

    _ensure_env_loaded()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    candidate_count = int(os.getenv("CANDIDATE_COUNT", "3"))
    styles_whitelist = _split_list(os.getenv("STYLES_WHITELIST"))
    log_path = Path(os.getenv("LOG_PATH", "logs/interactions.jsonl"))
    bandit_algo = os.getenv("BANDIT_ALGO", os.getenv("BANDIT_POLICY", "linucb"))

    return AppConfig(
        openai_api_key=openai_api_key,
        candidate_count=candidate_count,
        styles_whitelist=styles_whitelist,
        log_path=log_path,
        bandit_algo=bandit_algo,
    )
