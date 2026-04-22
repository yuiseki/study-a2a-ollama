"""Discovery デモ: 10 ナンセンス + 1 本命 + 1 ヒーロー を 1 プロセスで起動.

Hero に「来月の売上を予測せよ」というタスクを投げ、ヒーローが:
  1. 既知 URL 全てから Agent Card を取得（Direct Configuration discovery）
  2. LLM にタスク + カタログを渡して最適な相手を選ばせる
  3. 選ばれたエージェントに A2A で委譲
  4. 回答を回収してトップレベルに返す

...という流れを 1 ターミナルで観察できる。
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
    TaskStatusUpdateEvent,
    TextPart,
)

# package import のため src/ を sys.path に追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_discover.hero_server import (  # noqa: E402
    HeroExecutor,
    HOST as HERO_HOST,
    PORT as HERO_PORT,
    build_agent_card as build_hero_card,
)
from agent_discover.nonsense_server import (  # noqa: E402
    NonsenseExecutor,
    build_nonsense_card,
)
from agent_discover.revenue_oracle_server import (  # noqa: E402
    RevenueOracleExecutor,
    HOST as REVENUE_HOST,
    PORT as REVENUE_PORT,
    build_agent_card as build_revenue_card,
)

NONSENSE_TOPICS = [
    "食べ物",
    "音楽",
    "色",
    "動物",
    "季節",
    "映画",
    "飲み物",
    "スポーツ",
    "花",
    "乗り物",
]
NONSENSE_HOST = "127.0.0.1"
NONSENSE_BASE_PORT = 9001

TASK = "来月の売上を予測せよ"


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _text_from_message(m) -> str | None:
    if not m:
        return None
    for p in m.parts or []:
        root = getattr(p, "root", p)
        text = getattr(root, "text", None)
        if text:
            return text
    return None


async def start_server(executor, card, host: str, port: int):
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)
    config = uvicorn.Config(app.build(), host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.05)
    return server, task


async def run_client() -> None:
    url = f"http://{HERO_HOST}:{HERO_PORT}"
    async with httpx.AsyncClient(timeout=None) as httpx_client:
        cfg = ClientConfig(httpx_client=httpx_client, streaming=True)
        client = await ClientFactory.connect(agent=url, client_config=cfg)
        card = await client.get_card()
        print(f"{_now()} [client] Hero: {card.name}")
        print(f"{_now()} [client] タスク送信: {TASK}\n")

        msg = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=TASK))],
            message_id=str(uuid4()),
        )
        final: str | None = None
        async for item in client.send_message(msg):
            if isinstance(item, tuple):
                _t, upd = item
                if isinstance(upd, TaskStatusUpdateEvent):
                    state = upd.status.state.value if upd.status else "?"
                    text = _text_from_message(upd.status.message) if upd.status else None
                    print(f"{_now()} [hero ] state={state} text={text!r}")
                    if state == "completed":
                        final = text

        print()
        print(f"{_now()} [client] 最終結果:")
        print(final)


async def main() -> None:
    servers = []

    # 10 nonsense servers
    for i, topic in enumerate(NONSENSE_TOPICS):
        port = NONSENSE_BASE_PORT + i
        print(f"{_now()} [boot] nonsense '{topic}' :{port}")
        srv, t = await start_server(
            NonsenseExecutor(topic),
            build_nonsense_card(topic, NONSENSE_HOST, port),
            NONSENSE_HOST,
            port,
        )
        servers.append((srv, t))

    # revenue oracle
    print(f"{_now()} [boot] Revenue Forecast Oracle :{REVENUE_PORT}")
    srv, t = await start_server(
        RevenueOracleExecutor(),
        build_revenue_card(),
        REVENUE_HOST,
        REVENUE_PORT,
    )
    servers.append((srv, t))

    # hero
    print(f"{_now()} [boot] Discovery Hero :{HERO_PORT}")
    srv, t = await start_server(
        HeroExecutor(), build_hero_card(), HERO_HOST, HERO_PORT
    )
    servers.append((srv, t))

    print(f"{_now()} [boot] all 12 servers ready\n")

    try:
        await run_client()
    finally:
        for s, _ in servers:
            s.should_exit = True
        for _, t in servers:
            try:
                await t
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
