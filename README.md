# Online Conversation Optimizer


MVP that generates N reply candidates, annotates features, exposes action propensities, and learns online with a contextual bandit. Safe-by-default, evaluation-ready (IPS/DR logs).


## Quickstart

### Environment variables

| name | default | description |
| --- | --- | --- |
| `OPENAI_API_KEY` | _required for live generation_ | API key for OpenAI Responses API. Omit to use deterministic fallback. |
| `CANDIDATE_COUNT` | `3` | Default number of candidates per turn. Overridden by request `N`. |
| `STYLES_WHITELIST` | catalog styles | Comma-separated subset of styles permitted for generation. |
| `BANDIT_ALGO` | `linucb` | Choose `linucb` or `lints` for action selection. |
| `LOG_PATH` | `logs/interactions.jsonl` | Secondary JSONL stream used by the legacy logger. |
| `SAFETY_MIN_SCORE` | `0.2` | Minimum safety score before a candidate is rewritten/dropped. |

Run the API:

```bash
export OPENAI_API_KEY=your_key_here  # optional
pip install fastapi uvicorn numpy httpx pytest ruff jinja2 python-multipart
uvicorn src.app:app --reload
```

### REST API

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

### curl examples

Request a turn:

```bash
TURN=$(curl -s -X POST http://localhost:8000/turn \
  -H "Content-Type: application/json" \
  -d '{
        "history": ["集中できませんでした"],
        "user_utterance": "今日はどう動く？",
        "session_id": "demo"
      }')
echo "$TURN" | jq
```

Send feedback using values returned from the turn call:

```bash
SESSION=$(echo "$TURN" | jq -r '.session_id')
TURN_ID=$(echo "$TURN" | jq -r '.turn_id')
IDX=$(echo "$TURN" | jq -r '.chosen_idx')

curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"turn_id\": \"$TURN_ID\", \"chosen_idx\": $IDX, \"reward\": 0.6}"
```

## Logs

Each turn appends to `logs/turns-YYYYMMDD.jsonl` with previews, features, bandit metadata and rewards.

Run `python scripts/quick_report.py` to see aggregate reward, style win rates, exploration, and propensity stats.

## Web UI

- Start the API (`uvicorn src.app:app --reload`) and open <http://localhost:8000/ui>.
- 外部CDNに依存しないため、オフライン環境でも動作します。
- 候補数や報酬スライダーを調整して候補生成→カードをクリックするとフィードバック（latency 付き）が送信されます。
- ヘッダーの「会話をクリア」「新しいセッション」で履歴を即リセットできます。
- 右上の簡易メトリクスは `/metrics` を定期ポーリングして更新されます。
- **Screenshot placeholder:** _Add UI screenshot here_

## How to extend

- **Switch bandits**: set `BANDIT_ALGO=lints` (Thompson sampling) or `linucb` (upper-confidence bound) to change exploration behaviour without code changes.
- **Add new styles**: append entries to `prompts/11_styles_catalog.md` and include matching rules in `src/generation/generator.py` fallback prompts to ensure offline coverage.
- **Integrate alternate models**: tweak `DEFAULT_MODEL_NAME` or override `CandidateGenerator` while keeping the JSON contract (`text`, `style`, `features`).
- **Custom safety rules**: expand `src/safety/guard.py` with domain-specific heuristics or plug-in classifiers before the bandit sees candidates.
