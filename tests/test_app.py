from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def test_turn_and_feedback(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "interactions.jsonl"))
    monkeypatch.setenv("CANDIDATE_COUNT", "2")

    if "src.app" in sys.modules:
        importlib.invalidate_caches()
        sys.modules.pop("src.app")

    app_module = importlib.import_module("src.app")
    client = TestClient(app_module.app)

    payload = {
        "history": ["昨日は集中できませんでした。"],
        "user_utterance": "今日はどう進めればいい？",
        "session_id": "test-session",
    }
    response = client.post("/turn", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test-session"
    assert isinstance(data["turn_id"], str)
    assert "reply" in data and data["reply"]
    assert data["chosen_idx"] in {0, 1}
    assert 0.0 <= data["propensity"] <= 1.0
    assert "debug" in data

    feedback_payload = {
        "session_id": data["session_id"],
        "turn_id": data["turn_id"],
        "chosen_idx": data["chosen_idx"],
        "reward": 0.5,
    }
    feedback_response = client.post("/feedback", json=feedback_payload)
    assert feedback_response.status_code == 200

    log_path = Path(os.environ["LOG_PATH"])
    assert log_path.exists()
    with log_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    assert len(lines) == 2  # turn + feedback entries
