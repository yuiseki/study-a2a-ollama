"""Human Oracle (:8004) に A2A で問い合わせるデモ.

用途: `INPUT_REQUIRED` ではなく、独立した Agent Card を持つ人間ノードに
Task を流す設計が成立することを確認する。別ターミナルで
`human_oracle/human_server.py` を起動しておいてから実行する。
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import uuid4

import httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.types import (
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TextPart,
)

HUMAN_URL = "http://127.0.0.1:8004"
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


async def main() -> None:
    async with httpx.AsyncClient(timeout=None) as httpx_client:
        config = ClientConfig(httpx_client=httpx_client, streaming=True)
        client = await ClientFactory.connect(agent=HUMAN_URL, client_config=config)
        card = await client.get_card()
        print(f"{_now()} [card] {card.name}: {card.description}")
        print(f"{_now()} [card] skills: {[s.id for s in card.skills]}\n")

        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=QUERY))],
            message_id=str(uuid4()),
        )
        print(f"{_now()} [send] {QUERY}\n")

        final_text: str | None = None
        async for item in client.send_message(message):
            if isinstance(item, tuple):
                task, update = item
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


if __name__ == "__main__":
    asyncio.run(main())
