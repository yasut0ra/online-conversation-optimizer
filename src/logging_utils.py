"""Utilities for writing interaction logs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .types import Candidate, InteractionLogRecord


def _candidate_to_dict(candidate: Candidate) -> Dict[str, Any]:
    return {
        "text": candidate.text,
        "style": candidate.style,
        "features": candidate.features,
    }


@dataclass
class JsonlInteractionLogger:
    """Append-only JSONL logger for interaction data."""

    path: Path

    def log(self, record: InteractionLogRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "context_hash": record.context_hash,
            "candidates": [_candidate_to_dict(c) for c in record.candidates],
            "chosen_idx": record.chosen_idx,
            "propensity": record.propensity,
            "reward": record.reward,
            "features": record.features,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
