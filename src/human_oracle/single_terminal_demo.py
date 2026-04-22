"""Human Oracle サーバーと問い合わせクライアントを同一プロセスで起動するデモ.

scripts/human.sh から呼ばれる前提。別ターミナルを開かずに以下が 1 コマンドで完結:

  1. Human Oracle A2A サーバーを起動（:8004）
  2. 起動完了を待って demo_ask_human 相当のクライアントを走らせる
  3. サーバーが stdin で応答を求めるので、このターミナルで人間が 1 行返す
  4. クライアントが `completed` イベントを受け取って終了
  5. サーバーも停止してプロセスが終わる
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import uuid4

import httpx
import uvicorn
from a2a.client import ClientConfig, ClientFactory
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TextPart,
)

# 既存のサーバー実装・Agent Card 構築ロジックを流用（同一ディレクトリの兄弟モジュール）
from human_server import HOST, PORT, HumanExecutor, build_agent_card

QUERY = "チャレンジ: A2A プロトコルを一言で表現するとしたら何と言いますか？"


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _text_from_message(message) -> str | None:
    if not message:
        return None
    for p in message.parts or []:
        root = getattr(p, "root", p)
        text = getattr(root, "text", None)
        if text:
            return text
    return None


async def build_server_task() -> tuple[uvicorn.Server, asyncio.Task]:
    request_handler = DefaultRequestHandler(
        agent_executor=HumanExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=build_agent_card(),
        http_handler=request_handler,
    )
    config = uvicorn.Config(
        app.build(),
        host=HOST,
        port=PORT,
        log_level="warning",  # クライアント出力と混ざらないよう抑える
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    # サーバー起動完了まで待つ
    for _ in range(120):
        if server.started:
            break
        await asyncio.sleep(0.05)
    return server, task


async def run_client() -> None:
    async with httpx.AsyncClient(timeout=None) as httpx_client:
        config = ClientConfig(httpx_client=httpx_client, streaming=True)
        client = await ClientFactory.connect(
            agent=f"http://{HOST}:{PORT}", client_config=config
        )
        card = await client.get_card()
        print(f"{_now()} [card] {card.name}: {card.description}")
        print(f"{_now()} [card] skills: {[s.id for s in card.skills]}")
        print(f"{_now()} [send] {QUERY}\n")

        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=QUERY))],
            message_id=str(uuid4()),
        )
        final_text: str | None = None
        async for item in client.send_message(message):
            if isinstance(item, tuple):
                _task, update = item
                if isinstance(update, TaskStatusUpdateEvent):
                    state = update.status.state.value if update.status else "?"
                    text = _text_from_message(update.status.message) if update.status else None
                    print(f"{_now()} [evt ] state={state} text={text!r}")
                    if state == "completed":
                        final_text = text
                elif isinstance(update, TaskArtifactUpdateEvent):
                    print(f"{_now()} [evt ] artifact={update.artifact.artifact_id}")

        print()
        if final_text is not None:
            print(f"{_now()} [done] human's answer: {final_text}")
        else:
            print(f"{_now()} [done] (no completed state observed)")


async def main() -> None:
    server, server_task = await build_server_task()
    try:
        await run_client()
    finally:
        server.should_exit = True
        await server_task


if __name__ == "__main__":
    asyncio.run(main())
