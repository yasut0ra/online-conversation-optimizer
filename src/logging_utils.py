"""Utilities for writing interaction logs."""

from __future__ import annotations

import json
import os
import platform
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .types import Candidate, InteractionLogRecord

_APPEND_LOCK = threading.Lock()


def _candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "text": candidate.text,
        "style": candidate.style,
        "features": candidate.features,
    }


def _candidate_preview(candidate: Any) -> dict[str, Any]:
    text = ""
    style = "unknown"
    features: dict[str, Any] = {}
    if isinstance(candidate, Candidate):
        text = candidate.text
        style = candidate.style
        features = candidate.features
    elif isinstance(candidate, dict):
        text = str(candidate.get("text", ""))
        style = str(candidate.get("style", "unknown"))
        features = candidate.get("features") or {}
    else:
        text = str(candidate)
    return {
        "style": style,
        "text_preview": text[:120],
        "features": features,
    }


def _build_env_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    try:
        import fastapi

        versions["fastapi"] = fastapi.__version__
    except Exception:  # pragma: no cover - optional dependency
        pass
    try:
        import uvicorn

        versions["uvicorn"] = uvicorn.__version__
    except Exception:  # pragma: no cover - optional dependency
        pass
    return versions


ENV_VERSIONS = _build_env_versions()


def log_turn(session_id: str, turn_id: str, payload: dict[str, Any]) -> None:
    """Append a single turn entry to a date-partitioned JSONL file."""

    date_str = datetime.utcnow().strftime("%Y%m%d")
    log_path = Path("logs") / f"turns-{date_str}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = payload.get("candidates") or []
    if isinstance(candidates, Iterable) and not isinstance(candidates, (str, bytes)):
        candidates_preview = [_candidate_preview(c) for c in candidates]
    else:
        candidates_preview = []

    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "session_id": session_id,
        "turn_id": turn_id,
        "context_hash": payload.get("context_hash"),
        "candidates": candidates_preview,
        "chosen_idx": payload.get("chosen_idx"),
        "propensity": payload.get("propensity"),
        "reward": payload.get("reward"),
        "features": payload.get("features"),
        "bandit_algo": payload.get("bandit_algo")
        or os.getenv("BANDIT_ALGO")
        or os.getenv("BANDIT_POLICY"),
        "env_versions": ENV_VERSIONS,
    }
    if "phase" in payload:
        entry["phase"] = payload["phase"]

    with _APPEND_LOCK:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


@dataclass
class JsonlInteractionLogger:
    """Append-only JSONL logger for interaction data."""

    path: Path

    def log(self, record: InteractionLogRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "context_hash": record.context_hash,
            "session_id": record.session_id,
            "turn_id": record.turn_id,
            "candidates": [_candidate_to_dict(c) for c in record.candidates],
            "chosen_idx": record.chosen_idx,
            "propensity": record.propensity,
            "reward": record.reward,
            "features": record.features,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
