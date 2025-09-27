from __future__ import annotations

import importlib
import json
import os
import re
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

    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert metrics["turn_count"] >= 1


def test_api_turn_and_feedback(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "interactions.jsonl"))
    monkeypatch.setenv("CANDIDATE_COUNT", "2")

    if "src.app" in sys.modules:
        importlib.invalidate_caches()
        sys.modules.pop("src.app")

    app_module = importlib.import_module("src.app")
    client = TestClient(app_module.app)

    form_data = {
        "history_json": "[]",
        "user_utterance": "今日はどう動く？",
        "session_id": "ui-session",
        "candidate_count": "2",
    }
    response = client.post("/api/turn", data=form_data)
    assert response.status_code == 200
    html = response.text
    assert "candidate-card" in html

    session_match = re.search(r'data-session-id="([^"]+)"', html)
    turn_match = re.search(r'data-turn-id="([^"]+)"', html)
    candidate_match = re.search(r"data-candidate='([^']+)'", html)
    assert session_match and turn_match and candidate_match

    session_id = session_match.group(1)
    turn_id = turn_match.group(1)
    candidate_payload = json.loads(candidate_match.group(1))

    feedback_payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "chosen_idx": candidate_payload["index"],
        "reward": 1.0,
        "latency_ms": 42,
        "continued": True,
    }
    feedback_response = client.post("/api/feedback", json=feedback_payload)
    assert feedback_response.status_code == 200
    assert feedback_response.json()["status"] == "ok"
