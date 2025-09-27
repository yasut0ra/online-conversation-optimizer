"""Utilities for computing turn metrics from logs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

DEFAULT_LOG_DIR = Path("logs")


def _load_latest_records(log_dir: Path) -> list[dict[str, Any]]:
    files = sorted(log_dir.glob("turns-*.jsonl"))
    if not files:
        return []
    latest = files[-1]
    records: list[dict[str, Any]] = []
    with latest.open("r", encoding="utf-8") as handle:
        for line in handle:
            content = line.strip()
            if not content:
                continue
            try:
                records.append(json.loads(content))
            except json.JSONDecodeError:
                continue
    return records


def compute_metrics(log_dir: Path | None = None) -> dict[str, Any]:
    """Return aggregate metrics using the latest JSONL log.

    The returned dictionary contains keys: turn_count, avg_reward,
    style_win_rates, exploration_rate, propensity_mean, propensity_std.
    """

    directory = log_dir or DEFAULT_LOG_DIR
    records = _load_latest_records(directory)
    by_turn: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    for record in records:
        key = (record.get("session_id"), record.get("turn_id"))
        by_turn[key] = record

    final_records = list(by_turn.values())
    total = len(final_records)
    if total == 0:
        return {
            "turn_count": 0,
            "avg_reward": None,
            "style_win_rates": {},
            "exploration_rate": 0.0,
            "propensity_mean": None,
            "propensity_std": None,
        }

    rewards = [r["reward"] for r in final_records if r.get("reward") is not None]
    avg_reward = mean(rewards) if rewards else None

    prop_values = [r["propensity"] for r in final_records if r.get("propensity") is not None]
    prop_mean = mean(prop_values) if prop_values else None
    prop_std = pstdev(prop_values) if len(prop_values) > 1 else 0.0 if prop_values else None

    style_counter: Counter[str] = Counter()
    for record in final_records:
        candidates = record.get("candidates") or []
        chosen_idx = record.get("chosen_idx")
        if not isinstance(candidates, list) or chosen_idx is None:
            continue
        if 0 <= chosen_idx < len(candidates):
            style = candidates[chosen_idx].get("style", "unknown")
            style_counter[style] += 1

    style_win_rates = (
        {style: count / total for style, count in style_counter.items()}
        if style_counter
        else {}
    )

    unique_indices = {
        record.get("chosen_idx")
        for record in final_records
        if record.get("chosen_idx") is not None
    }
    exploration_rate = (len(unique_indices) / total) if total else 0.0

    return {
        "turn_count": total,
        "avg_reward": avg_reward,
        "style_win_rates": style_win_rates,
        "exploration_rate": exploration_rate,
        "propensity_mean": prop_mean,
        "propensity_std": prop_std,
    }


__all__ = ["compute_metrics"]
