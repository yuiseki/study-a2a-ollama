"""両 A2A サーバーに挨拶を送信して応答を確認するクライアント.

1. Agent Card を取得して表示
2. 同じ挨拶を両者に送り、最終応答を比較

a2a-sdk の新しい `ClientFactory.connect()` を使った実装.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, Task, TextPart

AGENTS = {
    "LangGraph (丁寧)": "http://127.0.0.1:8001",
    "Strands (タメ口)": "http://127.0.0.1:8002",
}
QUERY = "こんにちは。あなたは誰ですか？"


def _text_from_parts(parts) -> str | None:
    for p in parts or []:
        root = getattr(p, "root", p)
        text = getattr(root, "text", None)
        if text:
            return text
    return None


def _extract_text(item) -> str | None:
    """client.send_message() が yield する要素から応答テキストを抽出."""
    if isinstance(item, Message):
        return _text_from_parts(item.parts)
    if isinstance(item, tuple):
        task = item[0]
        if isinstance(task, Task):
            if task.artifacts:
                for art in task.artifacts:
                    text = _text_from_parts(art.parts)
                    if text:
                        return text
            if task.status and task.status.message:
                return _text_from_parts(task.status.message.parts)
    return None


async def ping_agent(label: str, base_url: str, config: ClientConfig) -> None:
    print(f"\n=== {label} @ {base_url} ===")
    client = await ClientFactory.connect(agent=base_url, client_config=config)
    card = await client.get_card()
    print(f"[card] name      = {card.name}")
    print(f"[card] skills    = {[s.id for s in card.skills]}")
    print(f"[card] streaming = {card.capabilities.streaming}")
    print(f"[send] {QUERY}")

    message = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text=QUERY))],
        message_id=str(uuid4()),
    )
    final: str | None = None
    async for item in client.send_message(message):
        text = _extract_text(item)
        if text:
            final = text
    print(f"[recv] {final}")


async def main() -> None:
    async with httpx.AsyncClient(timeout=180.0) as httpx_client:
        config = ClientConfig(httpx_client=httpx_client, streaming=True)
        for label, url in AGENTS.items():
            try:
                await ping_agent(label, url, config)
            except Exception as e:
                print(f"[error] {label}: {e!r}")


if __name__ == "__main__":
    asyncio.run(main())
