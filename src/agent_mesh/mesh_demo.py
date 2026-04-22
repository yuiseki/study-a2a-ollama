"""Agent A + Agent B + Human Oracle を同一プロセスで動かすマルチホップ委譲デモ.

Flow:

  client ──▶ Agent A (:8005) ──▶ Agent B (:8006) ──▶ Human Oracle (:8004) ──▶ human

Agent A が 1 桁の足し算を出題し、Agent B がそれを Human Oracle に委託し、
人間が答えをタイプし、答えが A2A 越しに A まで届く。

scripts/mesh.sh から呼ばれる前提。

各エージェントは Agent Card を持ち、お互いの内部実装を知らない。これが A2A の
「マルチホップ委譲」の最小実証。
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
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

# 兄弟モジュール / human_oracle ディレクトリからの import
# （src/ は sys.path に入っているので、ディレクトリ名を package として参照）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_mesh.coordinator import (  # noqa: E402
    CoordinatorExecutor,
    HOST as COORD_HOST,
    PORT as COORD_PORT,
    build_agent_card as build_coord_card,
)
from agent_mesh.delegator import (  # noqa: E402
    DelegatorExecutor,
    HOST as DELG_HOST,
    PORT as DELG_PORT,
    build_agent_card as build_delg_card,
)
from human_oracle.human_server import (  # noqa: E402
    HumanExecutor,
    HOST as HUMAN_HOST,
    PORT as HUMAN_PORT,
    build_agent_card as build_human_card,
)


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


async def start_server(
    executor, build_card, host: str, port: int
) -> tuple[uvicorn.Server, asyncio.Task]:
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=build_card(),
        http_handler=request_handler,
    )
    config = uvicorn.Config(
        app.build(), host=host, port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.05)
    return server, task


async def run_client() -> None:
    """トップレベル（スクリプト）が Agent A に問い合わせる."""
    coord_url = f"http://{COORD_HOST}:{COORD_PORT}"
    async with httpx.AsyncClient(timeout=None) as httpx_client:
        config = ClientConfig(httpx_client=httpx_client, streaming=True)
        client = await ClientFactory.connect(agent=coord_url, client_config=config)
        card = await client.get_card()
        print(f"{_now()} [client] Agent A: {card.name} / skills={[s.id for s in card.skills]}")
        print(f"{_now()} [client] 「計算して」と依頼します\n")

        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text="計算して"))],
            message_id=str(uuid4()),
        )
        final_text: str | None = None
        async for item in client.send_message(message):
            if isinstance(item, tuple):
                _t, update = item
                if isinstance(update, TaskStatusUpdateEvent):
                    state = update.status.state.value if update.status else "?"
                    text = _text_from_message(update.status.message) if update.status else None
                    print(f"{_now()} [evt A] state={state} text={text!r}")
                    if state == "completed":
                        final_text = text
                elif isinstance(update, TaskArtifactUpdateEvent):
                    print(f"{_now()} [evt A] artifact={update.artifact.artifact_id}")

        print()
        print(f"{_now()} [client] 最終結果: {final_text}")


async def main() -> None:
    # 3 つのサーバーを立ち上げ
    print(f"{_now()} [boot] starting Human Oracle ...")
    human_srv, human_task = await start_server(
        HumanExecutor(), build_human_card, HUMAN_HOST, HUMAN_PORT
    )
    print(f"{_now()} [boot] starting Agent B (Delegator) ...")
    delg_srv, delg_task = await start_server(
        DelegatorExecutor(), build_delg_card, DELG_HOST, DELG_PORT
    )
    print(f"{_now()} [boot] starting Agent A (Coordinator) ...")
    coord_srv, coord_task = await start_server(
        CoordinatorExecutor(), build_coord_card, COORD_HOST, COORD_PORT
    )
    print(f"{_now()} [boot] all servers ready\n")

    try:
        await run_client()
    finally:
        for srv in (coord_srv, delg_srv, human_srv):
            srv.should_exit = True
        for t in (coord_task, delg_task, human_task):
            try:
                await t
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
