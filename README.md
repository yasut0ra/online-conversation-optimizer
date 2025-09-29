# Online Conversation Optimizer

MVPとしての会話最適化システムです。各ターンで複数候補を生成し、特徴量を付与し、コンテキストバンディットで即時学習します。安全性チェックやログ蓄積がデフォルトで有効になっており、IPS / Doubly Robust といった評価にもすぐ対応できます。

## クイックスタート

### 環境変数

| 変数名 | 既定値 | 説明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 必須 (リアル生成時) | OpenAI Chat Completions API のキー。未設定の場合は決定論的なフォールバックのみ利用。|
| `CANDIDATE_COUNT` | `3` | 各ターンで生成する候補数。リクエストの `N` が優先。|
| `STYLES_WHITELIST` | スタイルカタログ全体 | 生成を許可するスタイルIDのカンマ区切りリスト。|
| `BANDIT_ALGO` | `linucb` | バンディットのアルゴリズム。`linucb` または `lints`。|
| `LOG_PATH` | `logs/interactions.jsonl` | 旧来のJSONLロガーが書き込むファイルパス。|
| `SAFETY_MIN_SCORE` | `0.2` | 安全スコアが閾値を下回った場合はリライト／除外。|

プロンプトは Markdown カタログから順番に連結され、空振りした場合でも内部の `DEFAULT_SYSTEM_PROMPT` にフォールバックするため、Chat Completions へのリクエストが常に安定します。

API サーバーを起動:

```bash
export OPENAI_API_KEY=your_key_here  # 任意
pip install fastapi uvicorn numpy httpx pytest ruff jinja2 python-multipart
uvicorn src.app:app --reload
```

## REST API

### `POST /turn`

リクエスト例:

```json
{
  "history": ["以前のやり取り"],
  "user_utterance": "今どうすればいい？",
  "N": 3,
  "session_id": "optional-session"
}
```

レスポンス例:

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

```json
{
  "session_id": "optional-session",
  "turn_id": "generated-turn-id",
  "chosen_idx": 1,
  "reward": 0.5
}
```

## curl での利用例

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

続けてフィードバックを送信:

```bash
SESSION=$(echo "$TURN" | jq -r '.session_id')
TURN_ID=$(echo "$TURN" | jq -r '.turn_id')
IDX=$(echo "$TURN" | jq -r '.chosen_idx')

curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"turn_id\": \"$TURN_ID\", \"chosen_idx\": $IDX, \"reward\": 0.6}"
```

## ログ

各ターンは `logs/turns-YYYYMMDD.jsonl` に追記され、候補プレビュー／特徴量／バンディットのメタデータ／報酬がまとまります。

`python scripts/quick_report.py` を実行すると、累積報酬やスタイル別勝率、探索率、propensity の統計を即座に確認できます。

## Web UI

- `uvicorn src.app:app --reload` を起動し、<http://localhost:8000/ui> を開く。
- 外部 CDN に依存しないのでオフライン環境でも UI が動作。
- 候補数や報酬スライダーを調整しながら候補を生成し、カードをクリックすると latency 付きでフィードバック送信。
- ヘッダーの「会話をクリア」「新しいセッション」で即リセット。
- 右上の簡易メトリクスは `/metrics` を定期ポーリングして更新。
- **スクリーンショット差し込み予定**: _後で画像を追加してください_

## 拡張方法

- **バンディットを切り替える**: 環境変数 `BANDIT_ALGO` を `lints` (Thompson Sampling) か `linucb` (UCB) に変更。
- **スタイルを追加する**: `prompts/11_styles_catalog.md` にスタイルを追加し、オフライン fallback を保つなら `src/generation/generator.py` のテンプレートへ文面を追記。
- **セーフティネットを調整する**: `src/generation/generator.py` 内の `DEFAULT_SYSTEM_PROMPT` を編集して、自前プロンプトが空になっても指示が欠落しないようにする。
- **モデルを差し替える**: `DEFAULT_MODEL_NAME` を変更するか `CandidateGenerator` を継承／差し替えて JSON 契約 (`text`, `style`, `features`) を維持。
- **安全ルールを強化する**: `src/safety/guard.py` にドメイン固有のルールや分類器を追加し、バンディットに渡す前にフィルタリング。
