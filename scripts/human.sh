#!/bin/bash
# human デモ: 人間を A2A ノードとして動かす（1 ターミナル版）.
#
# Human Oracle サーバーと問い合わせクライアントを同一プロセスで起動する。
# 別ターミナルを開く必要は無い。このターミナルに質問が表示されるので、
# 人間として 1 行の応答を入力して Enter を押せばタスクが completed になる。
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python src/human_oracle/single_terminal_demo.py
