#!/bin/bash
# task デモ: 疑似調査タスクを起動し、SSE で状態遷移を観察する.
set -euo pipefail
cd "$(dirname "$0")/.."

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

echo "[start] slow_task_server (:8003)"
uv run python src/task_lifecycle/slow_task_server.py &
pids+=($!)

for _ in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8003/.well-known/agent-card.json >/dev/null 2>&1; then break; fi
  sleep 0.3
done

echo "[ready] running stream observer"
uv run python src/task_lifecycle/stream_observer.py
