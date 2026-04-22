#!/bin/bash
# hello デモ: LangGraph + Strands の 2 サーバーを起動してクライアントで挨拶を投げる.
set -euo pipefail
cd "$(dirname "$0")/.."

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

# モデルを事前ウォームアップ（初回ロードの時間を排除）
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

# サーバー起動待ち（Agent Card が取得できるまで最大 30 秒）
for url in http://127.0.0.1:8001/.well-known/agent-card.json http://127.0.0.1:8002/.well-known/agent-card.json; do
  for _ in $(seq 1 60); do
    if curl -sf "$url" >/dev/null 2>&1; then break; fi
    sleep 0.5
  done
done

echo "[ready] running client"
uv run python src/greeting/client.py
