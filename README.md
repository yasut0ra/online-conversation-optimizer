# Online Conversation Optimizer


MVP that generates N reply candidates, annotates features, exposes action propensities, and learns online with a contextual bandit. Safe-by-default, evaluation-ready (IPS/DR logs).


## Run (pseudo)
```bash
export OPENAI_API_KEY=...
uvicorn src.app:app --reload
```

## REST API

### `POST /turn`

Request body
```json
{
  "history": ["以前のやり取り"],
  "user_utterance": "今どうすればいい？",
  "N": 3,
  "session_id": "optional-session"
}
```

Response body
```json
{
  "session_id": "optional-session",
  "turn_id": "generated-turn-id",
  "reply": "...",
  "chosen_idx": 1,
  "propensity": 0.47,
  "debug": {
    "scores": [0.12, 0.47, 0.38],
    "styles": ["empathetic", "logical", "coach"]
  }
}
```

### `POST /feedback`

Request body
```json
{
  "session_id": "optional-session",
  "turn_id": "generated-turn-id",
  "chosen_idx": 1,
  "reward": 0.5
}
```

### curl example

```bash
curl -X POST http://localhost:8000/turn \
  -H "Content-Type: application/json" \
  -d '{"history": ["集中できませんでした"], "user_utterance": "今日はどう動く？", "session_id": "demo"}'

curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "turn_id": "<turn-id-from-turn>", "chosen_idx": 0, "reward": 0.6}'
```

## Logs

Each turn appends to `logs/turns-YYYYMMDD.jsonl` with previews, features, bandit metadata and rewards.

Run `python scripts/quick_report.py` to see aggregate reward, style win rates, exploration, and propensity stats.
