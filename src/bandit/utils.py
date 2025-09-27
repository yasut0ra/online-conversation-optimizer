"""Utility helpers for bandit implementations."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np


def get_env_float(name: str, default: float) -> float:
    """Parse an environment variable as float with fallback."""

    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def ensure_1d(array: np.ndarray) -> np.ndarray:
    """Return a contiguous 1-D float64 array."""

    arr = np.asarray(array, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected 1-D array")
    return np.ascontiguousarray(arr, dtype=float)


def ensure_2d(array: np.ndarray) -> np.ndarray:
    """Return a contiguous 2-D float64 array."""

    arr = np.asarray(array, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected 2-D array")
    return np.ascontiguousarray(arr, dtype=float)


def softmax(scores: np.ndarray, beta: float | None = None) -> np.ndarray:
    """Stable softmax with optional temperature scaling."""

    scores = ensure_1d(scores)
    if beta is None:
        beta = get_env_float("SOFTMAX_BETA", 1.0)
    scaled = scores * float(beta)
    scaled -= np.max(scaled)
    exps = np.exp(scaled)
    total = np.sum(exps)
    if not np.isfinite(total) or total <= 0.0:
        return np.ones_like(scores) / scores.size
    return exps / total


def context_hash(payload: Any) -> str:
    """Create a deterministic hash for a JSON-serialisable payload."""

    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

