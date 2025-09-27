from __future__ import annotations

import json
from datetime import datetime

from src.logging_utils import log_turn
from src.types import Candidate


def test_log_turn_creates_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    session_id = "sess"
    turn_id = "turn"
    candidates = [
        Candidate(text="テストメッセージです", style="empathetic", features={}),
        Candidate(text="二つ目の候補", style="logical", features={}),
    ]

    payload = {
        "context_hash": "hash",
        "candidates": candidates,
        "chosen_idx": 0,
        "propensity": 0.5,
        "reward": None,
        "features": {},
        "bandit_algo": "linucb",
    }

    log_turn(session_id, turn_id, payload)

    date_str = datetime.utcnow().strftime("%Y%m%d")
    log_path = tmp_path / "logs" / f"turns-{date_str}.jsonl"
    assert log_path.exists()

    line = log_path.read_text(encoding="utf-8").strip()
    record = json.loads(line)

    assert record["session_id"] == session_id
    assert record["turn_id"] == turn_id
    assert record["candidates"][0]["text_preview"] == "テストメッセージです"
    assert len(record["candidates"][0]["text_preview"]) <= 120
