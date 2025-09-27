from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def test_turn_and_feedback(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "interactions.jsonl"))
    monkeypatch.setenv("BANDIT_STATE_PATH", str(tmp_path / "linucb.json"))
    monkeypatch.setenv("CANDIDATE_COUNT", "2")

    if "src.app" in sys.modules:
        importlib.invalidate_caches()
        sys.modules.pop("src.app")

    app_module = importlib.import_module("src.app")
    client = TestClient(app_module.app)

    payload = {
        "messages": [
            {"role": "user", "content": "仕事の優先順位が分からなくて悩んでいます"}
        ]
    }
    response = client.post("/turn", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["chosen_idx"] in {0, 1}
    assert len(data["candidates"]) == 2
    assert len(data["propensities"]) == 2

    context_hash = data["context_hash"]
    feedback_response = client.post(
        "/feedback", json={"context_hash": context_hash, "reward": 0.5}
    )
    assert feedback_response.status_code == 200

    log_path = Path(os.environ["LOG_PATH"])
    assert log_path.exists()
    with log_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    assert len(lines) == 2  # turn + feedback entries
