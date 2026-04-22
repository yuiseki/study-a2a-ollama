#!/bin/bash
# cross_talk デモ: Strands -> LangGraph -> Strands のバケツリレーを一発実行.
#
# greeting の 2 サーバーを起動し、モデルを事前ウォームアップしてから
# cross_talk.py を実行する。
set -euo pipefail
cd "$(dirname "$0")/.."

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

# モデル事前ウォームアップ
echo "[warmup] pulling models into memory"
curl -s http://localhost:11434/api/generate -d '{"model":"gemma3:4b","prompt":"hi","stream":false,"options":{"num_predict":1}}' >/dev/null &
curl -s http://localhost:11434/api/generate -d '{"model":"qwen3.5:4b","prompt":"hi","stream":false,"options":{"num_predict":1}}' >/dev/null &
wait

echo "[start] LangGraph server (:8001)"
uv run python src/greeting/langgraph_server.py &
pids+=($!)

echo "[start] Strands server (:8002)"
uv run python src/greeting/strands_server.py &
pids+=($!)

# サーバー起動待ち
for url in http://127.0.0.1:8001/.well-known/agent-card.json http://127.0.0.1:8002/.well-known/agent-card.json; do
  for _ in $(seq 1 60); do
    if curl -sf "$url" >/dev/null 2>&1; then break; fi
    sleep 0.5
  done
done

echo "[ready] running cross_talk"
uv run python src/greeting/cross_talk.py
