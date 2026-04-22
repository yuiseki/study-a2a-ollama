#!/bin/bash
# agent_mesh デモ: Agent A → Agent B → Human のマルチホップ委譲を 1 ターミナルで実演.
#
# Agent A が LLM で 1 桁の足し算を出題し、Agent B が LLM で人間への依頼文に清書し、
# Human Oracle が stdin で答えを受け取る。答えが A2A 越しに A まで届く。
set -euo pipefail
cd "$(dirname "$0")/.."

# モデル事前ウォームアップ
echo "[warmup] pulling models into memory"
curl -s http://localhost:11434/api/generate -d '{"model":"gemma3:4b","prompt":"hi","stream":false,"options":{"num_predict":1}}' >/dev/null &
curl -s http://localhost:11434/api/generate -d '{"model":"qwen3.5:4b","prompt":"hi","stream":false,"options":{"num_predict":1}}' >/dev/null &
wait

uv run python src/agent_mesh/mesh_demo.py
