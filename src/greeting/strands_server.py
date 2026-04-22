"""Strands Agents + Ollama を A2A サーバーとして公開する（:8002）.

Strands の `A2AServer` はエージェントをそのまま A2A HTTP エンドポイントとして
wrap してくれるため、LangGraph 側より短く書ける。
"""
from __future__ import annotations

from strands import Agent
from strands.models.ollama import OllamaModel
from strands.multiagent.a2a import A2AServer

HOST = "127.0.0.1"
PORT = 8002
MODEL = "qwen3.5:4b"
SYSTEM_PROMPT = (
    "あなたはフレンドリーでカジュアルな日本語で応答するアシスタントです。"
    "タメ口・友達口調で、語尾は「〜だよ」「〜だね」を使ってください。"
    "返答は 1〜2 文の短い挨拶に留めてください。"
)


def main() -> None:
    model = OllamaModel(
        host="http://localhost:11434",
        model_id=MODEL,
        max_tokens=256,  # 挨拶だけなので短く打ち切る
        temperature=0.3,
        # qwen3.5 の thinking を無効化。Ollama API の top-level `think` パラメータに
        # そのまま渡される（Strands OllamaModel は additional_args を request dict に
        # スプレッドする実装）。
        additional_args={"think": False},
    )
    agent = Agent(
        model=model,
        name="Strands Casual Greeter",
        description="Strands + Ollama qwen3:0.6b で動くタメ口の挨拶エージェント",
        system_prompt=SYSTEM_PROMPT,
    )
    server = A2AServer(agent=agent, host=HOST, port=PORT)
    server.serve()


if __name__ == "__main__":
    main()
