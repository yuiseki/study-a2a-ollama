#!/bin/bash
# agent_discover デモ: 10 ナンセンス + 1 売上予測 + 1 ヒーロー を同プロセスで起動し、
# LLM が Agent Card カタログから適切な相手を選ぶ様子を観察する.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[warmup] pulling model into memory"
curl -s http://localhost:11434/api/generate \
  -d '{"model":"gemma3:4b","prompt":"hi","stream":false,"options":{"num_predict":1}}' \
  >/dev/null

uv run python src/agent_discover/discover_demo.py
